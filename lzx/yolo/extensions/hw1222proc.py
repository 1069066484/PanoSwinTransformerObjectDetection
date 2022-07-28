import shutil
import os


def run(top='.'):
    curr_ch = 'a'
    d = {}
    for fd in os.listdir(top):
        fd_new = ''
        for c in fd:
            if c in "0123456789_":
                fd_new += c
            else:
                if c not in d:
                    d[c] = curr_ch
                    c = curr_ch
                    curr_ch = chr(ord(curr_ch) + 1)
                    if c == 'z':
                        curr_ch = 'A'
                else:
                    c = d[c]
                fd_new += c
        fd_old_ = os.path.join(top, fd)
        fd_new_ = os.path.join(top, fd_new)
        shutil.move(fd_old_, fd_new_)
        print("{} -> {}".format(fd_old_, fd_new_))
    print("dict:", d)




import string


d = {}
pref = 'a3q'
valid_letters = string.ascii_letters + string.digits + '_.'

def run2(top='.'):
    for fd in os.listdir(top):
        if fd.endswith(".py") or fd.endswith('.out'): continue
        fd_new = ''
        try:
            for c in fd:
                if c in valid_letters:
                    fd_new += c
                else:
                    if c not in d:
                        d[c] = pref + str(len(d))
                        c = d[c]
                    else:
                        c = d[c]
                    fd_new += c
            fd_old_ = os.path.join(top, fd)
            fd_new_ = os.path.join(top, fd_new)
            shutil.move(fd_old_, fd_new_)
            print("{} -> {}".format(fd_old_, fd_new_))
        except:
            print("{} failed".format(fd))



def run_final(top):
    run2(top)
    if 1:
        for folder in os.listdir(top):
            try:
                folder = os.path.join(top, folder)
                run2(folder)
            except:
                print("{} failed".format(folder))
    print("dict:", d)


if __name__=='__main__':
    run_final('.')

