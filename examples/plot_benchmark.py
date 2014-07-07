"""
Given some output from benchmarks.py, produce a graph.

Pass the filename of the output to plot as the first argument.

Requires the following non-standard Python modules:
 - matplotlib
 - parse
"""

import matplotlib, parse, sys, time
matplotlib.use('SVG')
import matplotlib.pyplot as plt

def parse_platform(index, lines):
    current_index = index
    platform = lines[index][9:].strip()
    x = []
    y = []
    for line in lines[index+1:]:
        if line.find("PLATFORM:") == 0:
            return current_index, platform, x, y
        if line.find("(") == 0:
            format = "({x}, {y})"
            parsed = parse.parse(format, line)
            x.append(int(parsed['x'].strip()))
            y.append(float(parsed['y'].strip()))
        current_index += 1
    return current_index, platform, x, y

def get_data(f):
    platforms = {}
    lines = f.readlines()
    next_index = 0
    current_index = 0
    title = None
    for line in lines:
        if line.find("OPERATION:") == 0:
            title = line[11:].strip()
            break
        current_index += 1

    if not title:
        return None, None

    next_index = current_index
    for line in lines[next_index:]:
        if line.find("PLATFORM:") == 0:
            next_index, platform, x, y = parse_platform(current_index, lines)
            platforms[platform] = (x, y)
        current_index += 1
    
    return title, platforms
            

with open(sys.argv[1]) as f:
    title, platforms = get_data(f)

ax = plt.subplot(1,1,1)
ax.set_xscale('log')
ax.set_yscale('log')

plt.title(title)
plt.xlabel('Size')
plt.ylabel('Execution time (s)')

for platform in platforms.keys():
    ax.plot(platforms[platform][0], platforms[platform][1], "-o",
            label=platform)

handles, labels = ax.get_legend_handles_labels()
import operator
hl = sorted(zip(handles, labels),
            key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)
leg = ax.legend(handles2, labels2, fancybox=True, loc=2, prop={'size':11})
leg.get_frame().set_alpha(0.5)

save_as = time.strftime("%Y%m%d-%H%M") + " - " + title + ".svg"
plt.savefig(save_as)

print("Saved figure to %s" % save_as)

