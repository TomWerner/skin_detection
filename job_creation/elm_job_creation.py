import sys


def create_job(args):
    output_template = """
#!/bin/sh

# This selects which queue
#$ -q UI
# One node. 1-16 cores on smp
#$ -pe smp 16
# Make the folder first
#$ -o /Users/twrner/outputs
#$ -e /Users/twrner/errors

cd ~/skin_detection/op_elm

~/anaconda/bin/python elm_trainer.py %s
""" % (" ".join(args[1:]))
    file = open("/Users/twrner/jobs/train_elm_%s.job" % (args[0]), 'w')
    file.write(output_template)
    file.close()


def parse_arg(flag, sys_args, default):
    if flag in sys_args:
        return sys_args[sys_args.index(flag) + 1]
    return default


if __name__ == '__main__':
    if "-h" in sys.argv:
        print("python elm_trainer.py <job tag> <filename> <batch size> [(lin|sigm|tanh)-neuron-###]")
    else:

        create_job(sys.argv[1:])