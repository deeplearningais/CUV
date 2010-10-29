#!/usr/bin/python2.6
import os
import numpy as np
import pyrbmview as mainwin
from plotwin import Ui_PlotWindow
from simpleplotwin import Ui_SimplePlotWindow
import simpleplotwin
import PyQt4
from PyQt4.QtCore import QObject, SIGNAL
from PyQt4 import QtCore, QtGui

import matplotlib
matplotlib.use('QT4Agg')
from matplotlib import pyplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure



class simpleplotwin( QtGui.QFrame, Ui_SimplePlotWindow ):
    def __init__( self, name, mainwindow ):
        QtGui.QFrame.__init__( self )
        self.name = name
        self.mainwindow = mainwindow
        self.dpi = 100

    def _name( self ):
        return str(self.name)

    def loadSettings(self, settings):
        self.logscale.setChecked( settings.value("logscale", True).toBool() )
        rect = settings.value("geometry", QtCore.QRect(10,10,100,100)).toRect()
        self.setGeometry( rect )
        self.wantAutomaticUpdate.setChecked(settings.value("wantAutomaticUpdate", True).toBool())
        print self.geometry()

    def writeSettings(self, settings):
        print "Writing settings for ", self._name()
        settings.beginGroup(self._name())
        settings.setValue( "name",           self.name )
        settings.setValue( "type",           "simpleplot" )
        settings.setValue( "logscale",       self.logscale.isChecked() )
        settings.setValue( "geometry",       self.geometry())
        settings.setValue( "wantAutomaticUpdate", self.wantAutomaticUpdate.isChecked() )
        print self.geometry()
        settings.endGroup()

    def setupUi( self, pw ):
        self.main_frame = self
        super( simpleplotwin, self ).setupUi( pw )
        self.setWindowTitle(self.name)

        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_subplot(111)

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        self.connect(self.updateButton, SIGNAL('clicked()'), self._actionOnUpdateClicked)
        
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.mpl_toolbar)
        self.data = np.zeros( (1,28) )

    def pull_device( self ):
        self.data = self.mainwindow.pyro.getmat( self._name() )

    def pull_host( self ):
        self.data = self.mainwindow.pyro.getmat_host( self._name() )

    def _actionOnUpdateClicked( self ):
        self.pull_device()
        self.on_draw()

    def on_draw(self):
        if self.data == None: 
            print "No new data for", self._name()
            return
        self.axes.clear()

        if len(self.data.shape)>1:
            if len(self.data)>20:
                data=[]
                data.append(np.convolve(self.data,10*[1./10],mode="valid"))
                data.append(np.convolve(self.data,10*[1./10],mode="valid"))
            else:
                data=self.data
            self.axes.plot(data[0],label="training:%.2f"%(data[0,-1]*100))
            self.axes.plot(data[1],label="testing:%.2f"%(data[1,-1]*100))
            self.axes.legend()
        else:
            if len(self.data)>20:
                data=np.convolve(self.data,10*[1./10],mode="valid")
            else:
                data=self.data
            self.axes.plot(data)
            self.axes.set_axis_on()

        if self.logscale.isChecked():
            self.axes.set_yscale( 'log' )
        self.canvas.draw()

class matshowwin( QtGui.QFrame, Ui_PlotWindow ):
    def __init__( self, name, mainwindow ):
        QtGui.QFrame.__init__( self )
        self.name = name
        self.mainwindow = mainwindow
        self.dpi = 100

    def _name( self ):
        return self.name + "->" + str(self.minibatchIdx.value())

    def loadSettings(self, settings):
        val, ok = settings.value( "minibatchIdx" ).toInt()
        self.minibatchIdx.setValue([0,val][ok])
        rect = settings.value("geometry", QtCore.QRect(10,10,100,100)).toRect()
        self.wantAutomaticUpdate.setChecked(settings.value("wantAutomaticUpdate", False).toBool())
        self.setGeometry( rect )

    def writeSettings(self, settings):
        self.mainwindow.statusbar.showMessage( "Writing settings for %s" % self._name() )
        settings.beginGroup(self._name())
        settings.setValue( "name", self.name )
        settings.setValue( "type", "matshow" )
        settings.setValue( "minibatchIdx", self.minibatchIdx.value() )
        settings.setValue( "geometry",       self.geometry())
        settings.setValue( "wantAutomaticUpdate", self.wantAutomaticUpdate.isChecked() )
        settings.endGroup()

    def setupUi( self, pw ):
        self.main_frame = self
        super( matshowwin, self ).setupUi( pw )
        self.setWindowTitle(self.name)
        self.minibatchIdx.setMaximum(10000)

        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_subplot(111)

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        self.connect(self.updateButton, SIGNAL('clicked()'), self._actionOnUpdateClicked)
        
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.mpl_toolbar)
        self.data = np.zeros( (28,28) )
        self.firstrun = True

    def pull_device( self ):
        self.data = self.mainwindow.pyro.getmat( self._name() )

    def pull_host( self ):
        self.data = self.mainwindow.pyro.getmat_host( self._name() )

    def _actionOnUpdateClicked( self ):
        self.pull_device()
        self.on_draw()

    def on_draw(self):
        self.axes.clear()
        if self.data == None: 
            print "No new data for", self._name()
            return
        self.p = self.axes.matshow(self.data,cmap=pyplot.cm.gray)
        if self.firstrun:
            self.colorbar = self.axes.figure.colorbar( self.p )
            self.firstrun = False
        self.colorbar.update_bruteforce(self.p)
        self.colorbar.draw_all()
        self.canvas.draw()

class pyrbmview( QtGui.QMainWindow, mainwin.Ui_pyRBMView ):
    def __init__(self, workdir):
        QtGui.QMainWindow.__init__(self)
        self.workdir = workdir
        self.setupUi(self)
        self.__setupConnections()
        self.statusbar.showMessage("Loading settings")
        self.loadSettings()

    def _actionConnect( self ):
        import Pyro.core
        Pyro.core.initClient()
        self.statusbar.showMessage("opening pyro.url in `%s'" % self.workdir)
        url = "fail"
        with open(os.path.join( self.workdir, "pyro.url"), "r") as f:
           url = f.readline()
        self.statusbar.showMessage("connecting to `%s'..." % url)
        self.pyro = Pyro.core.getProxyForURI( url )

        self.statusbar.showMessage("Loading RBM configuration...")
        self.rbmcfg = self.pyro.getcfg()
        import socket
        host = socket.gethostname()
        self.treeWidget.setHeaderLabels([host + "::" + self.workdir])

        tlid = 0
        self.treeWidget.clear()
        item_0 = QtGui.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(tlid).setText(0, "params")
        for k in sorted(self.rbmcfg.keys()):
            item_sub = QtGui.QTreeWidgetItem( item_0 )
            item_sub.setText(0, str(k) + "=" + str( self.rbmcfg[k] ))
        tlid += 1

        item_0 = QtGui.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(tlid).setText(0, "reconstruction_error")
        tlid += 1
        item_0 = QtGui.QTreeWidgetItem(self.treeWidget)
        self.treeWidget.topLevelItem(tlid).setText(0, "Dataset")
        tlid += 1
        item_sub = QtGui.QTreeWidgetItem( item_0 )
        item_sub.setText(0, "mean")
        item_sub = QtGui.QTreeWidgetItem( item_0 )
        item_sub.setText(0, "range")
        item_sub = QtGui.QTreeWidgetItem( item_0 )
        item_sub.setText(0, "min")
        item_sub = QtGui.QTreeWidgetItem( item_0 )
        item_sub.setText(0, "max")
        item_sub = QtGui.QTreeWidgetItem( item_0 )
        item_sub.setText(0, "std")
        #states = [ "afterweightupdate", "aftersave", "originals" ]
        states = self.rbmcfg["states"]

        for l in xrange( len(self.rbmcfg["l_size"]) ):
            item_l = QtGui.QTreeWidgetItem(self.treeWidget)
            self.treeWidget.topLevelItem(tlid).setText(0, "Layer%d"%l)

            for s in states:
                item_sub = QtGui.QTreeWidgetItem(item_l)
                item_sub.setText(0, s)
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "Act")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "Sub Act")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "PChain")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "Sub PChain")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "AllWeights")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "Weights")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "BiasLo")
                item_subsub = QtGui.QTreeWidgetItem( item_sub )
                item_subsub.setText(0, "BiasHi")
            tlid += 1
    def _actionContinuousUpdatesChanged(self):
        checked = self.actionContinuous_Updates.isChecked()
        if checked:
            self._actionStartContinuousUpdates()
        else:
            self._actionStopContinuousUpdates()
    def _actionStartContinuousUpdates(self):
        self.statusbar.showMessage("starting continuous updates")
        self.timer = QtCore.QTimer(self)
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"),self._actionUpdate_All)
        self.timer.start( 2500 )
    def _actionStopContinuousUpdates(self):
        self.statusbar.showMessage("stopping continuous updates")
        self.timer.stop()

    def setupUi( self, pyRBMView ):
        mainwin.Ui_pyRBMView.setupUi( self, pyRBMView )
        self.connect(self.actionContinuous_Updates, QtCore.SIGNAL("toggled(bool)") , self._actionContinuousUpdatesChanged)
        self.connect(self.actionSave_Config, QtCore.SIGNAL("activated()") , self.writeSettings)
        self.connect(self.actionConnect, QtCore.SIGNAL("activated()") , self._actionConnect)

    
    def writeSettings(self):
        settings = QtCore.QSettings("ais.uni-bonn.de", "rbmview")
        settings.beginGroup("/RBMView")
        settings.setValue("/geometry/pos", self.pos())
        settings.setValue("/geometry/size", self.size())
        settings.setValue("/geometry/mainSplitter", self.splitter.saveState())
        settings.beginGroup("/MDIsettings")
        settings.remove("")
        for w in self.mdiArea.subWindowList():
            w.widget().writeSettings(settings)
        settings.endGroup()
        settings.endGroup()

    def loadSettings(self):
        settings = QtCore.QSettings("ais.uni-bonn.de", "rbmview")
        settings.beginGroup("/RBMView")
        size = settings.value("/geometry/size", QtCore.QSize(800,600)).toSize()
        pos  = settings.value("/geometry/pos", QtCore.QPoint(100,100)).toPoint()
        self.splitter.restoreState(settings.value("/geometry/mainSplitter").toByteArray())
        settings.beginGroup("/MDIsettings")
        for w in settings.childGroups():
            settings.beginGroup(w)
            self._addSubWinBySettings(settings)
            settings.endGroup()
        settings.endGroup()
        settings.endGroup()
        self.move(pos)
        self.resize(size)
    
    def closeEvent( self, event ):
        #self.writeSettings()
        event.accept()

    def _addPlotWin(self,widget):
        subwin = self.mdiArea.addSubWindow(widget)
        widget.setupUi( widget )
        subwin.resize(widget.size())
        subwin.show()

    def __setupConnections( self ):
        self.connect(self.actionShowSubWin, QtCore.SIGNAL("triggered()") , self._actionShowSubWin)
        self.connect(self.actionUpdate_All, QtCore.SIGNAL("triggered()") , self._actionUpdate_All)

    def _actionUpdate_All(self):
        if not"pyro" in self.__dict__: assert False, "No connection!"
        WL = filter(lambda x:x.widget().wantAutomaticUpdate.isChecked(), self.mdiArea.subWindowList())
        L = map( lambda x:x.widget()._name(), WL)
        self.pyro.pullmats(L)
        map( lambda x:x.widget().pull_host(), WL )
        map( lambda x:x.widget().on_draw(), WL )


    def _actionShowSubWin(self):
        it = self.treeWidget.currentItem()
        par = it.parent()
        if not par:
            self._addSubWinByStr(it.text(0))
        else:
            par2 = par.parent()
            if par2:
                s = str( par2.text( 0 ) ) + "->" + str(par.text(0)) + "->" + str(it.text(0))
                self._addSubWinByStr(s)
            else:
                s = str(par.text(0)) + "->" + str(it.text(0))
                self._addSubWinByStr(s)

    def _addSubWinBySettings(self,settings):
        name = settings.value("name").toString()
        type = settings.value("type").toString()
        if type == "matshow":
            pw = matshowwin(name, self)
            self._addPlotWin(pw)
            pw.loadSettings(settings)
        if type == "simpleplot":
            pw = simpleplotwin(name, self)
            self._addPlotWin(pw)
            pw.loadSettings(settings)


    def _addSubWinByStr(self,name):
        if name in [ "reconstruction_error"]:
            self._addPlotWin( simpleplotwin(name,self) )
        else:
            self._addPlotWin( matshowwin(name,self) )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "Need Working Directory!"
    app = QtGui.QApplication(sys.argv)
    ui = pyrbmview(sys.argv[1])
    ui.show()
    sys.exit(app.exec_())


