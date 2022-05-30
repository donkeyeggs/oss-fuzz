import os
import shutil
import random

import functions as FUNC


class STYLE:
    DEFAULT = 0
    BOLD = 1
    ITALIC = 3
    UNDERLINE = 4
    ANTIWHITE = 7


class COLOR:
    DEFAULT = 39
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36
    WHITE = 37
    LIGHTBLACK_EX = 90
    LIGHTRED_EX = 91
    LIGHTGREEN_EX = 92
    LIGHTYELLOW_EX = 93
    LIGHTBLUE_EX = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX = 96
    LIGHTWHITE_EX = 97


class BG_COLOR:
    DEFAULT = 49
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    PURPLE = 45
    CYAN = 46
    WHITE = 47
    LIGHTBLACK_EX = 100
    LIGHTRED_EX = 101
    LIGHTGREEN_EX = 102
    LIGHTYELLOW_EX = 103
    LIGHTBLUE_EX = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX = 106
    LIGHTWHITE_EX = 107


class Font(object):
    def __init__(self, text, style=STYLE.BOLD, color=COLOR.GREEN, bg_color=BG_COLOR.DEFAULT):
        self.color = color
        self.style = style
        self.bg_color = bg_color
        self.text = text

    def __str__(self):
        return self.__repr__()
        #return f"FONT({self.text}, style = {self.style}, color = {self.color}, bg_color = {self.bg_color})"

    def __repr__(self):
        ret = []
        for line in self.text.split('\n'):
            front = f'\033[{self.style};{self.color};{self.bg_color}m'
            inline = line
            back = f"\033[0m"
            ret.append(front + inline + back)
        return '\n'.join(ret)


class Log(object):
    def __init__(self,
                 project_name="(none)",
                 headline="DEBUG >>>",
                 log_dir=None,
                 console=False
                 ):
        self.nothing = (FUNC.PLATFORM()=="linux")
        self._project_name = project_name
        self.PROJECT_NAME = project_name
        self._headline = headline
        self.project_name = Font(f"({project_name})", style=STYLE.BOLD, color=COLOR.PURPLE)
        self.headline = Font(headline, style=STYLE.BOLD, color=COLOR.GREEN)
        self.console = console
        if log_dir is None:
            if FUNC.PLATFORM() == "linux":
                self.log_dir = "/src/database"
            elif FUNC.PLATFORM() == "windows":
                self.log_dir = "./database"
        else:
            self.log_dir = log_dir
        if self.nothing:
            return
        self.series = FUNC.now_time()
        self.log_dir = os.path.join(self.log_dir, project_name, self.series)
        self.couted = False
        self.lastFile = None
        if not os.path.exists(self.log_dir) and not self.console:
            os.makedirs(self.log_dir)

    def logHead(self, goon=False):
        if self.nothing:
            return
        print(Font(f" == now in project ({self._project_name}) == ", STYLE.BOLD, COLOR.PURPLE))
        if not self.console and not goon:
            self.lastFile = FUNC.new_name()

    def log(self, *args, **kwargs):
        if self.nothing:
            return
        print(self.headline, end='')

        print(*args, **kwargs)

        if not self.console and self.lastFile is not None:
            outdir = os.path.join(self.log_dir, self.lastFile)
            with open(outdir, 'a', encoding='utf-8') as fout:
                print(*args, **kwargs, file=fout)

    def log_empty(self):
        if self.nothing:
            return
        print("")
        if not self.console:
            outdir = os.path.join(self.log_dir, self.lastFile)
            with open(outdir, 'a', encoding='utf-8') as fout:
                print("",file=fout)

    def logEnd(self):
        if self.nothing:
            return
        lastfile = os.path.join(self.log_dir, self.lastFile)
        if not self.console:
            shutil.copy(lastfile, os.path.join(self.log_dir, "__can_read__.txt"))

    def __del__(self):
        if self.nothing:
            return
        print(Font(f"== log out ,file saved in {self.series}==", STYLE.BOLD, COLOR.BLUE))

