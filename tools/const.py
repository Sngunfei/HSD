# -*- encoding: utf-8 -*-

import platform

PAGERANK = "PageRank"
EIGEN_CENTRALITY = "EigenCentrality"

System = platform.system()
LinuxRootPath = "/home/master/2019/songyunfei/workspace/py/HSD"
WindowsRootPath = "G:\pyworkspace\HSD"

HierarchyLiunxPathTemplate = "/home/master/2019/songyunfei/workspace/py/HSD/data/hierarchy/{}.layers"
HierarchyWindowsPathTemplate = "G:\pyworkspace\HSD\data\hierarchy\{}.layers"

CLASS_INFO = {
    "mkarate": 34,
    "barbell": 8,
}
