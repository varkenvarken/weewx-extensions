from json import load, dump

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1","-1")
    parser.add_argument("--in2","-2")
    parser.add_argument("--filename","-f")
    args = parser.parse_args()

    with open(args.in1) as in1:
        with open(args.in2) as in2:
            data1 = load(in1)
            data2 = load(in2)
            data = data1 + data2
            with open(args.filename, "w") as fp:
                dump(data, fp)
