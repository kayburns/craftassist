import os
import sys

dir_test = os.path.dirname(__file__)
dir_craftassist = os.path.join(dir_test, "..")
dir_root = os.path.join(dir_craftassist, "..")

sys.path.append(dir_craftassist)
sys.path.append(dir_root)
sys.path.insert(0, dir_test)  # insert 0 so that Agent is pulled from here

print("sys path {}".format(sys.path))
