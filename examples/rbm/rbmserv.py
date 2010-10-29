import os
import Pyro.core

import threading
from threading import Thread

#Pyro.config.PYRO_HOST = "131.220.7.90"
Pyro.config.PYRO_PORT = 7768



class RBMServ(Pyro.core.ObjBase):
        def __init__(self, rbm):
            Pyro.core.ObjBase.__init__(self)
            self.rbm = rbm

        def getcfg(self):
            return self.rbm.cfg.get_serialization_obj()
        def pullmats(self, L):
            L = [str(x) for x in L]
            e = threading.Event()
            e.clear()
            self.rbm.request_mats(L, e)
            e.wait(1.)
        def getmat_host(self, name):
            name = str(name)
            if str(name) in self.rbm.pulled:
                ret = self.rbm.pulled[name]
                del self.rbm.pulled[name]
                return ret
            else: return None
        def getmat(self, name):
            #print("getmat")
            name = str(name)
            e = threading.Event()
            e.clear()
            self.rbm.request_mat(name, e)
            e.wait(1.)
            if name in self.rbm.pulled:
                ret = self.rbm.pulled[name]
                del self.rbm.pulled[name]
                return ret
            else: return None

class RBMServPyroMain(Thread):
    def __init__(self,rbm,workdir):
        Thread.__init__(self)
        self.rbm = rbm
        self.workdir = workdir
        self.setDaemon(True)
        self.rbm_serv=None
    def run(self):
        Pyro.core.initServer()
        daemon=Pyro.core.Daemon()
        self.rbm_serv=RBMServ(self.rbm)
        uri=daemon.connect(self.rbm_serv,"rbmserv")
        self.rbm.pyro_daemon = daemon

        print "The daemon runs on port:",daemon.port
        print "The object's uri is:",uri

        with open(os.path.join(self.workdir, "pyro.url"), "w") as f:
            f.write(str(uri))

        daemon.requestLoop()



