import matplotlib.pyplot as plt
import argparse

arg_parser=argparse.ArgumentParser()
arg_parser.add_argument('--error', type=float)
arg_parser.add_argument('--logfile', type=str)


args=arg_parser.parse_args()
print(args)
error_lim=args.error
logfilen=args.logfile
#'current/logfile-29f6bf0ae7844543a34bbf21c9698794.log'

logfile=open(logfilen)
force_errors=[]
n_frames=0
for line in logfile:
    if 'std' in line:
        force_errors.append(float(line.strip().split()[-1]))
        n_frames+=1

print(force_errors)
plt.hlines(error_lim,0,n_frames,'r')
plt.plot(force_errors)
plt.xlabel('Frame')
plt.ylabel("Force stdev (meV / A)")
plt.show()