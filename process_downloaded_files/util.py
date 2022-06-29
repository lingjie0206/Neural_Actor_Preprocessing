
import fcntl
from pathlib import Path
class FileLock():
    def __init__(self, fpath):
        if fpath is Path:
            self.lock = fpath
        else:
            self.lock = Path(fpath)
        self.has_lock = True

    def __enter__(self):
        print('Locking file %s'%self.lock)
        self.f = open(self.lock.as_posix(),'w')
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            #self.f.write(hostname())
            self.f.flush()
        except IOError:
            print('Locking failed %s'%self.lock)
            self.has_lock = False
        return self
    def __exit__(self, type, value, traceback):
        if self.has_lock:
            print('Unlocking file %s'%self.lock)
            self.f.close()
            try:
                self.lock.unlink()
            except:
                pass
