import os
import platform
import sys

def do_help():
    print("Requires one argument:")
    print("  python -- to get the Python version")
    print("  boost  -- to get the Boost Python component name")

def main(argv):
    if len(argv) < 2:
        do_help()
        return os.EX_CONFIG

    if argv[1] == "python":
        print("%d.%d.%d" % (sys.version_info.major,
                            sys.version_info.minor,
                            sys.version_info.micro))
    elif argv[1] == "boost":
        if platform.system() == 'Linux':
            if platform.linux_distribution()[0] == 'Ubuntu':
                print("python-py%d%d" %
                      (sys.version_info.major,
                       sys.version_info.minor))
                return os.EX_OK

        if sys.version_info == 3:
            print("python3")
        else:
            print("python")
    else:
        do_help()
        return os.EX_CONFIG

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv))
