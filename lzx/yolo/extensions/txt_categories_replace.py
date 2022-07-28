try:
    from extensions.tt_split import d
    from extensions.merge_bbs import write_txt_single, read_txt_single
except:
    from tt_split import d
    from merge_bbs import write_txt_single, read_txt_single
import os
import argparse


names = ['ACdist', 'batteries', 'wirelesscabinet', 'heat-tubeac', 'ladderbatteries', 'generalcabinet', 'powercabinet', 'FSU', 'groundwire', 'ac', 'othercabinet', 'fansys', 'hangingac', 'groundbox', 'transformer', 'powerbox', 'Libatteries', 'unifiedcabinet', 'monitorbox', 'acexternal', 'cabinet', 'DCdistribution', 'ladderbattery']


def run(top_folder, from_cats, to_cat):
    assert all([n in names for n in from_cats + [to_cat]])
    print("{} - {}".format(from_cats, to_cat))
    to_cat = names.index(to_cat)
    from_cats = [names.index(cat) for cat in from_cats]
    print("{} - {}".format(from_cats, to_cat))
    n_files_checked = 0
    n_replaced = 0
    n_original = 0
    for root, dirs, files in os.walk(top_folder):
        for file in files:
            if not file.endswith(".txt"):
                continue
            try:
                n_files_checked += 1
                file = os.path.join(root, file)
                confidence, xyhw, cls = read_txt_single(file)
                for i in range(len(cls)):
                    if cls[i] == to_cat:
                        n_original += 1
                    elif cls[i] in from_cats:
                        n_replaced += 1
                        cls[i] = to_cat
                write_txt_single(file, xyhw, cls)
            except:
                print("Error processing {}, abort".format(file))
        if n_files_checked % 500 == 0:
            print("n_files_checked: {}, n_original: {}, n_replaced: {}".format(n_files_checked, n_original, n_replaced))
    print("n_files_checked: {}, n_original: {}, n_replaced: {}".format(n_files_checked, n_original, n_replaced))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_folder', type=str)
    args = parser.parse_args()
    # run("data_trainable1213", ["ladderbatteries", "Libatteries", "ladderbattery"], "batteries")
    run(args.top_folder, ["ladderbatteries", "Libatteries", "ladderbattery"], "batteries")

'''
python txt_categories_replace.py > data_trainable1213/txt_categories_replace.log 2>&1 &
'''
