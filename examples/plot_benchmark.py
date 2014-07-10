"""
Given some output from benchmarks.py, produce a graph.

Pass the filename of the output to plot as the first argument.

Requires the following non-standard Python modules:
 - matplotlib
 - parse
"""

import matplotlib, numpy as np, parse, sys, time
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
            
def no_mangling(platforms):
    return 'Execution time', 'log', 'log', 'Size', 'Execution time (s)', platforms

def add_mangling(platforms):
    extra_title = 'Add (Bandwidth)'
    xscale, yscale = 'linear', 'linear'
    xlabel = 'Size'
    ylabel = 'Memory Bandwidth (GB/s)'
    new_platforms = {}
    def compute_new_y(size, time):
        return (np.float32().itemsize * 3 * size) / (time*(10**9))
    for platform in platforms.keys():
        x = platforms[platform][0]
        y = platforms[platform][1]
        new_y = list(map(compute_new_y, x, y))
        new_platforms[platform] = (x, new_y)
    return extra_title, xscale, yscale, xlabel, ylabel, new_platforms

def gemv_mangling(platforms):
    extra_title = 'GEMV (Bandwidth)'
    xscale, yscale = 'linear', 'linear'
    xlabel = 'Size'
    ylabel = 'Memory Bandwidth (GB/s)'
    new_platforms = {}
    def compute_new_y(size, time):
        return (np.float32().itemsize * (size**2)) / (time * (10**9))
    for platform in platforms.keys():
        x = platforms[platform][0]
        y = platforms[platform][1]
        new_y = list(map(compute_new_y, x, y))
        new_platforms[platform] = (x, new_y)
    return extra_title, xscale, yscale, xlabel, ylabel, new_platforms

def gemm_mangling(platforms):
    extra_title = 'GEMM (FLOPs)'
    xscale, yscale = 'linear', 'linear'
    xlabel = 'Size'
    ylabel = 'GFLOP/s'
    new_platforms = {}
    def compute_new_y(size, time):
        return (2 * (size**3)) / (time * (10**9))
    for platform in platforms.keys():
        x = platforms[platform][0]
        y = platforms[platform][1]
        new_y = list(map(compute_new_y, x, y))
        new_platforms[platform] = (x, new_y)
    return extra_title, xscale, yscale, xlabel, ylabel, new_platforms

with open(sys.argv[1]) as f:
    title, platforms = get_data(f)

if len(sys.argv) > 2:
    if sys.argv[2] == 'add':
        mangling = add_mangling
    elif sys.argv[2] == 'gemv':
        mangling = gemv_mangling
    elif sys.argv[2] == 'gemm':
        mangling = gemm_mangling
else:
    mangling = no_mangling

extra_title, xscale, yscale, xlabel, ylabel, platforms = mangling(platforms)

fig, ax = plt.subplots()
ax.set_xscale(xscale)
ax.set_yscale(yscale)

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

for platform in platforms.keys():
    ax.plot(platforms[platform][0], platforms[platform][1], "-o",
            label=platform)

fig_ratio = 1.2
fig.set_size_inches(fig.get_size_inches()[1]/fig_ratio, fig.get_size_inches()[1])

handles, labels = ax.get_legend_handles_labels()
import operator
hl = sorted(zip(handles, labels),
            key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)
leg = ax.legend(handles2, labels2, fancybox=True, loc=2, prop={'size':11})
leg.get_frame().set_alpha(0.5)

save_as = time.strftime("%Y%m%d-%H%M") + " - " + title + ' - ' + extra_title + ".svg"
plt.savefig(save_as)

print("Saved figure to %s" % save_as)

