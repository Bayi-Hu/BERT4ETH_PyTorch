import numpy as np
import pickle as pkl
import argparse
import functools
import os

parser = argparse.ArgumentParser(description="Data Processing with PyTorch")
parser.add_argument("--phisher", type=bool, default=True, help="whether to include phisher detection dataset.")
parser.add_argument("--deanon", type=bool, default=True, help="whether to include de-anonymization dataset.")
parser.add_argument("--tornado", type=bool, default=True, help="whether to include tornado dataset.")
parser.add_argument("--data_dir", type=str, default="../data", help="data directory.")
parser.add_argument("--dataset", type=str, default=None, help="which dataset to use")
parser.add_argument("--bizdate", type=str, default='gas', help="the date of running experiments.")
parser.add_argument("--dup", type=str, default=True, help="whether to do transaction duplication")
args = parser.parse_args()

print("Duplication:", args.dup)


HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(",")

def cmp_udf(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0

def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0

def load_data(f_in, f_out):

    def _gen_seq_data(file_read, in_out_type: str):
        eoa2seq_out = {}
        while True:
            trans = file_read.readline()
            if trans == "":
                break
            record = trans.split(",")
            trans_hash = record[0]
            block_number = int(record[3])
            from_address = record[5]
            to_address = record[6]
            value = int(record[7]) / pow(10, 12)
            gas = int(record[8])
            gas_price = int(record[9])
            gas_fee = int(gas*gas_price)/ pow(10, 12)
            block_timestamp = int(record[11])
            if from_address == "" or to_address == "":
                print(f"Dataset error: transactions {trans_hash} does not have valid input/output address!")
                continue
            if in_out_type == "IN":
                from_address, to_address = to_address, from_address
            try:
                eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value,  in_out_type, 1, gas_fee])
            except:
                eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, in_out_type, 1, gas_fee]]
        return eoa2seq_out

    eoa2seq_in = _gen_seq_data(f_in, "IN")
    eoa2seq_out = _gen_seq_data(f_out, "OUT")

    return eoa2seq_in, eoa2seq_out


def seq_generation(eoa2seq_in, eoa2seq_out, dedpulicate=False):

    eoa_list = list(eoa2seq_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all = cnt_all + (trans[5] if dedpulicate else 1)
            # if cnt_all >= 5 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq


def seq_duplicate(eoa2seq_in, eoa2seq_out):

    def _duplication(sequence):
        eoa2seq_agg_out = {}
        for eoa in sequence.keys():
            if len(sequence[eoa]) >= 10000:
                continue
            seq_sorted = sorted(sequence[eoa], key=functools.cmp_to_key(cmp_udf))
            seq_tmp = [e.copy() for e in seq_sorted]
            for i in range(len(seq_tmp) - 1, 0, -1):
                l_acc = seq_tmp[i][0]  # latter
                f_acc = seq_tmp[i - 1][0]  # former
                l_time = int(seq_tmp[i][2])
                f_time = int(seq_tmp[i - 1][2])
                delta_time = l_time - f_time
                if f_acc != l_acc or delta_time > 86400 * 3:
                    continue
                # value add
                seq_tmp[i - 1][3] += seq_tmp[i][3]
                seq_tmp[i - 1][5] += seq_tmp[i][5]
                seq_tmp[i - 1][-1] += seq_tmp[i][-1]
                del seq_tmp[i]
            eoa2seq_agg_out[eoa] = seq_tmp

        return eoa2seq_agg_out

    eoa2seq_agg_out = _duplication(eoa2seq_out)
    eoa2seq_agg_in = _duplication(eoa2seq_in)

    eoa2seq_agg = seq_generation(eoa2seq_agg_in, eoa2seq_agg_out, dedpulicate=True)

    return eoa2seq_agg


def feature_bucketization(eoa2seq_agg):
    # how to split the amount

    for eoa in eoa2seq_agg.keys():
        seq = eoa2seq_agg[eoa]
        for trans in seq:
            amount = trans[3]
            cnt = trans[5]
            gas_fee = trans[-1]

            if amount == 0:
                amount_bucket = 1
            elif amount<= 591:
                amount_bucket = 2
            elif amount<= 6195:
                amount_bucket = 3
            elif amount <= 21255:
                amount_bucket = 4
            elif amount <= 50161:
                amount_bucket = 5
            elif amount <= 100120:
                amount_bucket = 6
            elif amount <= 208727:
                amount_bucket = 7
            elif amount <= 508961:
                amount_bucket = 8
            elif amount <= 1360574:
                amount_bucket = 9
            elif amount <= 6500000:
                amount_bucket = 10
            elif amount <= 143791433950:
                amount_bucket = 11
            else:
                amount_bucket = 12

            trans[3] = amount_bucket

            if cnt == 0:
                cnt_bucket = 0
            elif cnt == 1:
                cnt_bucket = 1
            elif cnt == 2:
                cnt_bucket = 2
            elif cnt == 3:
                cnt_bucket = 3
            elif cnt == 4:
                cnt_bucket = 4
            elif cnt == 5:
                cnt_bucket = 5
            elif cnt == 6:
                cnt_bucket = 6
            elif cnt == 7:
                cnt_bucket = 7
            elif 8 < cnt <= 10:
                cnt_bucket = 8
            elif 10 < cnt <= 20:
                cnt_bucket = 9
            else:
                cnt_bucket = 10

            trans[5] = cnt_bucket

            # gas fee
            if gas_fee == 0.0:
                gas_fee_buck = 1
            elif gas_fee <= 51:
                gas_fee_buck = 2
            elif gas_fee <= 126:
                gas_fee_buck = 3
            elif gas_fee <= 301:
                gas_fee_buck = 4
            elif gas_fee <= 500:
                gas_fee_buck = 5
            elif gas_fee <= 801:
                gas_fee_buck = 6
            elif gas_fee <= 1113:
                gas_fee_buck = 7
            elif gas_fee <= 1702:
                gas_fee_buck = 8
            elif gas_fee <= 2332:
                gas_fee_buck = 9
            elif gas_fee <= 3579:
                gas_fee_buck = 10
            elif gas_fee <= 45570:
                gas_fee_buck = 11
            else:
                gas_fee_buck = 12

            trans[-1] = gas_fee_buck

    return eoa2seq_agg


def main():

    f_in = open(os.path.abspath(os.path.join(args.data_dir, "normal_eoa_transaction_in_slice_1000K.csv")), "r")
    f_out = open(os.path.abspath( os.path.join(args.data_dir, "normal_eoa_transaction_out_slice_1000K.csv")), "r")
    print("Add normal account transactions.")
    eoa2seq_in, eoa2seq_out = load_data(f_in, f_out)
    if args.dup:
        eoa2seq_agg = seq_duplicate(eoa2seq_in, eoa2seq_out)
    else:
        eoa2seq_agg = seq_generation(eoa2seq_in, eoa2seq_out)

    if args.phisher:
        print("Add phishing..")
        phisher_f_in = open(os.path.abspath(os.path.join(args.data_dir, "phisher_transaction_in.csv")), "r")
        phisher_f_out = open(os.path.abspath(os.path.join(args.data_dir, "phisher_transaction_out.csv")), "r")
        phisher_eoa2seq_in, phisher_eoa2seq_out = load_data(phisher_f_in, phisher_f_out)
        if args.dup:
            phisher_eoa2seq_agg = seq_duplicate(phisher_eoa2seq_in, phisher_eoa2seq_out)
        else:
            phisher_eoa2seq_agg = seq_generation(phisher_eoa2seq_in, phisher_eoa2seq_out)
        eoa2seq_agg.update(phisher_eoa2seq_agg)
    if args.deanon:
        print("Add ENS..")
        dean_f_in = open(os.path.join(args.data_dir, "dean_trans_in_new.csv"), "r")
        dean_f_out = open(os.path.join(args.data_dir, "dean_trans_out_new.csv"), "r")
        dean_eoa2seq_in, dean_eoa2seq_out = load_data(dean_f_in, dean_f_out)
        if args.dup:
            dean_eoa2seq_agg = seq_duplicate(dean_eoa2seq_in, dean_eoa2seq_out)
        else:
            dean_eoa2seq_agg = seq_generation(dean_eoa2seq_in, dean_eoa2seq_out)
        eoa2seq_agg.update(dean_eoa2seq_agg)

    if args.tornado:
        print("Add tornado...")
        tornado_in = open(os.path.join(args.data_dir, "tornado_trans_in_removed.csv"), "r")
        tornado_out = open(os.path.join(args.data_dir, "tornado_trans_out_removed.csv"), "r")
        tornado_eoa2seq_in, tornado_eoa2seq_out = load_data(tornado_in, tornado_out)
        if args.dup:
            tornado_eoa2seq_agg = seq_duplicate(tornado_eoa2seq_in, tornado_eoa2seq_out)
        else:
            tornado_eoa2seq_agg = seq_generation(tornado_eoa2seq_in, tornado_eoa2seq_out)
        eoa2seq_agg.update(tornado_eoa2seq_agg)

    eoa2seq_agg = feature_bucketization(eoa2seq_agg)

    print("statistics:")
    length_list = []
    for eoa in eoa2seq_agg.keys():
        seq = eoa2seq_agg[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq #:", len(length_list))

    os.makedirs("../inter_data", exist_ok=True)

    with open("../inter_data/eoa2seq_" + args.bizdate + ".pkl", "wb") as f:
        pkl.dump(eoa2seq_agg, f)


if __name__ == '__main__':
    main()