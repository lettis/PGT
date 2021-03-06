#!/usr/bin/env python
import os
import sys
import platform
import tempfile
import logging
import cPickle
import copy
import operator
import hashlib
from glob import glob

try:
  import numpy
  import scipy
  import scipy.sparse
except:
  print "error: numpy/scipy not found. cannot do without it. sorry."
  sys.exit()

try:
  import pexpect
except:
  print "warning: pexpect not found. won't be able to run GROMACS binaries"

try:
  import matplotlib.pyplot as plt
  from matplotlib.font_manager import FontProperties
  import matplotlib
except:
  print "warning: matplotlib not found. won't be able to produce plots."

try:
  import networkx as nx
except:
  print "warning: networkx not found. won't be able to produce networks/graphs."


# set log-level to INFO
logging.basicConfig( level=logging.INFO )



#################
# definitions

#  atomic masses [u]; from GROMACS 'atommass.dat'
# additional values introduced (marked by comment)
atomMasses = {
 "H"   :    1.00790,
 "He"  :    4.00260,
 "Li"  :    6.94100,
 "Be"  :    9.01220,
 "B"   :   10.81100,
 "C"   :   12.01070,
  "CA" :   12.01070, # additional
 "N"   :   14.00670,
 "O"   :   15.99940,
 "F"   :   18.99840,
 "Ne"  :   20.17970,
 "Na"  :   22.98970,
 "Mg"  :   24.30500,
 "Al"  :   26.98150,
 "Si"  :   28.08550,
 "P"   :   30.97380,
 "S"   :   32.06500,
 "Cl"  :   35.45300,
 "Ar"  :   39.94800,
 "K"   :   39.09830,
 "Ca"  :   40.07800,
 "Sc"  :   44.95590,
 "Ti"  :   47.86700,
 "V"   :   50.94150,
 "Cr"  :   51.99610,
 "Mn"  :   54.93800,
 "Fe"  :   55.84500,
 "Co"  :   58.93320,
 "Ni"  :   58.69340,
 "Cu"  :   63.54600,
 "Zn"  :   65.39000,
 "Ga"  :   69.72300,
 "Ge"  :   72.64000,
 "As"  :   74.92160,
 "Se"  :   78.96000,
 "Br"  :   79.90400,
 "Kr"  :   83.80000,
 "Rb"  :   85.46780,
 "Sr"  :   87.62000,
 "Y"   :   88.90590,
 "Zr"  :   91.22400,
 "Nb"  :   92.90640,
 "Mo"  :   95.94000,
 "Tc"  :   98.00000,
 "Ru"  :  101.07000,
 "Rh"  :  102.90550,
 "Pd"  :  106.42000,
 "Ag"  :  107.86820,
 "Cd"  :  112.41100,
 "In"  :  114.81800,
 "Sn"  :  118.71000,
 "Sb"  :  121.76000,
 "Te"  :  127.60000,
 "I"   :  126.90450,
 "Xe"  :  131.29300,
 "Cs"  :  132.90550,
 "Ba"  :  137.32700,
 "La"  :  138.90550,
 "Ce"  :  140.11600,
 "Pr"  :  140.90770,
 "Nd"  :  144.24000,
 "Pm"  :  145.00000,
 "Sm"  :  150.36000,
 "Eu"  :  151.96400,
 "Gd"  :  157.25000,
 "Tb"  :  158.92530,
 "Dy"  :  162.50000,
 "Ho"  :  164.93030,
 "Er"  :  167.25900,
 "Tm"  :  168.93420,
 "Yb"  :  173.04000,
 "Lu"  :  174.96700,
 "Hf"  :  178.49000,
 "Ta"  :  180.94790,
 "W "  :  183.84000,
 "Re"  :  186.20700,
 "Os"  :  190.23000,
 "Ir"  :  192.21700,
 "Pt"  :  195.07800,
 "Au"  :  196.96650,
 "Hg"  :  200.59000,
 "Tl"  :  204.38330,
 "Pb"  :  207.20000,
 "Bi"  :  208.98040,
 "Po"  :  209.00000,
 "At"  :  210.00000,
 "Rn"  :  222.00000,
 "Fr"  :  223.00000,
 "Ra"  :  226.00000,
 "Ac"  :  227.00000,
 "Th"  :  232.03810,
 "Pa"  :  231.03590,
 "U "  :  238.02890,
 "Np"  :  237.00000,
 "Pu"  :  244.00000,
 "Am"  :  243.00000,
 "Cm"  :  247.00000,
 "Bk"  :  247.00000,
 "Cf"  :  251.00000,
 "Es"  :  252.00000,
 "Fm"  :  257.00000,
 "Md"  :  258.00000,
 "No"  :  259.00000,
 "Lr"  :  262.00000,
 "Rf"  :  261.00000,
 "Db"  :  262.00000,
 "Sg"  :  266.00000,
 "Bh"  :  264.00000,
 "Hs"  :  277.00000,
 "Mt"  :  268.00000
}


#################
# tools
def binning( valFrom, valTo, nBins ):
  """create list of lower bounds for binned intervall"""
  a = []
  binSize = float(valTo-valFrom) / nBins
  for i in range(nBins):
    a.append( valFrom + i*binSize )
  return a

def minMax( a1, a2 ):
  """order two values and return tuple (minValue, maxValue)"""
  a = [a1,a2]
  a.sort()
  return (a[0], a[1])


def splitList( a, nElems ):
  """split list into smaller lists with # = nElems elements per sublist"""
  lists = []
  buf = []
  i = 0
  for elem in a:
    i += 1
    buf.append( elem )
    if i == nElems:
      i=0
      lists.append( buf )
      buf = []
  if len(buf) > 0:
    lists.append( buf )
  return lists


def smoothen( xs, ys=None, width=100 ):
  """
  smoothen the given data (x- and y-values; given as lists) by
  generating new lists with the averages over #'width' datapoints
  """
  newXs = []
  newYs = []
  for i in range(width, len(xs) ):
    newXs.append( sum(xs[ i-width:i ]) / width )
    if ys:
      newYs.append( sum(ys[ i-width:i ]) / width )
  return (newXs, newYs)



def generateThetas( params, indexN, indexCA, indexNref, indexCAref ):
  """
  generate thetas [rad] between structures of a .gro-file and a reference
  defined by 'refIndex' (per default =1, i.e. the first structure in the file).
  thetas are taken as angle between vectors N->CA of both structures.
  """
  # read reference structure
  paramsRef = Params()
  paramsRef["input"] = params["reference"]
  ref = GroFile( paramsRef ).read().structures[0]
  # take vector N->CA (array index reduced from base1 to base0)
  refVec = ref.atoms[indexCAref-1].r - ref.atoms[indexNref-1].r
  # list of rot angles between structures
  class Thetas:
    def __init__(self, refVec):
      self.refVec = refVec
      self.thetas = []
    def appendTheta(self, struct):
      v = struct.atoms[indexCA-1].r - struct.atoms[indexN-1].r
      self.thetas.append( angleBetweenVectors(self.refVec, v) )
  thetas = Thetas( refVec )
  # calculate angles framewise
  GroFile(params).framewise( thetas.appendTheta )
  return thetas.thetas


def linewise( fh, func, ref ):
  """
  parse file linewise and perform 'func' on every line

  **arguments**

    - fh:     file handle to open file; will be closed after parsing
    - ref:    reference to outer object, see 'func'
    - func:   function to be exectuted per line
              must be of form
               < func( line, ref ) >
              'line' is the current line as string ( as read from fh.next() )
              'ref' is a reference to an outer structure/object,
              which can be altered by the function
              (e.g. a matrix containing the data parsed from a file)
  """
  try:
    while 1:
      try:
        line = fh.next()
        func(line, ref)
      except StopIteration:
        break
  finally:
    fh.close()


def blobDump( obj, filename ):
  """write serialized version of object to file"""
  fh = open( filename, "wb" )
  #cPickle.dump( obj, fh, cPickle.HIGHEST_PROTOCOL )
  cPickle.dump( obj, fh, 1 )
  fh.close()
  log = logging.getLogger( " blobDump " )
  log.info( "dumping '%s'-object to %s" % (obj.__class__.__name__, filename) )

def blobLoad( filename, generator=None ):
  """load object from file and return it"""
  log = logging.getLogger( " blobLoad " )

  try:
    fh = open( filename, "rb" )
  except IOError:
    if generator:
      log.info( "file not found: %s; will create object via generator function." % filename )
      obj = generator()
      blobDump( obj, filename )
      return obj
    else:
      raise
  obj = cPickle.load( fh )
  fh.close()
  log.info( "loading '%s'-object from %s" % (obj.__class__.__name__, filename) )
  return obj


def eulerAngles( R ):
  """
  returns vector of euler-angles calculated from given rotation matrix R.
  the angles will be given in 'x' or '313' convention.

  see `this tech report`__ for more information.

  __ http://www.astro.rug.nl/software/kapteyn/_downloads/attitude.pdf
  """
  r13 = R.item(0,2)
  r23 = R.item(1,2)
  r33 = R.item(2,2)
  r31 = R.item(2,0)
  r32 = R.item(2,1)
  phi   = numpy.math.atan2(r13, r23)
  theta = numpy.math.acos (r33)
  psi   = numpy.math.atan2(r31, -r32)
  return numpy.array( [phi, theta, psi] )

def angularDistance( R ):
  """
  calculate the angular distance (value between 0 and Pi) of a vector
  to its rotated counterpart after applying rotation matrix R to it.
  """
  v1 = numpy.array( [[1.0, 0.0, 0.0]] )
  v2 = R * numpy.matrix(v1.transpose())
  result = numpy.arccos( numpy.dot(v1,v2).item(0,0) )
  # sometimes value is NaN because of rounding errors.
  # treat these cases as 0 (which they usually should be)
  if not numpy.isnan(result):
    return result
  else:
    return 0.0


def angleBetweenVectors( v1, v2 ):
  normV1 = scipy.sqrt( scipy.dot(v1,v1) )
  normV2 = scipy.sqrt( scipy.dot(v2,v2) )
  return scipy.arccos( scipy.absolute(scipy.dot(v1,v2)) / normV1 / normV2 )


def rotFits( M, ref=None ):
  """
  read matrix M with cartesian 3D coordinates of N particles of form

  x1 y1 z1 x2 y2 z2 ... xN yN zN
  .
  .
  .
  x'1 y'1 z'1 x'2 y'2 z'2 ... x'N y'N z'N

  where every line is another structure of a trajectory.
  returns list of #lines 3x3 matrices giving the rotational fit of
  corresponding structure to either a given reference structure 'ref'
  (given in the same format as M with only one row) or to the first
  structure in the trajectory.
  least squares fit to 3D structure is based on [Sorkine2003].
  """
  nRows, nCols = M.shape
  def extractVs( M, r ):
    # split structure (i.e. current row) into list of 3D vectors
    vs = []
    for i in range(nCols / 3):
      x,y,z = M.item(r,3*i), M.item(r,3*i+1), M.item(r,3*i+2)
      vs.append( numpy.array([x,y,z]) )
    # center 3D vectors
    centroidStruct = numpy.zeros( 3 )
    for i in range(len(vs)):
      centroidStruct += vs[i]
    for i in range(len(vs)):
      vs[i] -= centroidStruct
    return vs

  # split ref-structure into list of 3D-vectors
  vs_ref = extractVs( M, 0 )

  # set up ref matrix for SVD fit
  Y = numpy.matrix(vs_ref).transpose()

  ## calculate rotation matrices
  M_fit = []
  for r in range(nRows):
    vs = extractVs( M, r )
    # set up matrix for SVD
    X = numpy.matrix(vs).transpose()
    # calculate 'variance' matrix for SVD
    S = X * numpy.identity( len(vs) ) * Y.transpose()
    U,_,Vt = numpy.linalg.svd( S )
    V = Vt.transpose()
    Ut = U.transpose()
    M_fit.append(  V * numpy.matrix( [[1,0,0], [0,1,0], [0,0,numpy.linalg.det(V*Ut)]] ) * Ut  )
  return M_fit


class Params(dict):
  """represents a parameter-list"""
  def __init__(self, *args, **kw):
    super(Params, self).__init__(*args, **kw)
    self.log = logging.getLogger( " Parameters " )
    self["PCA_BLOBS"] = []
    self["DIH_BLOBS"] = []

  def update(self, params):
    for key in params:
      self[key] = params[key]

  def setDefault(self, key, val):
    if key not in self.keys():
      self[key] = val

  def load(self, path="."):
    """search current directory for available data and load as default params"""
    self["PATH"] = path
    ndxFiles = glob( path + "/*.ndx" )
    if len(ndxFiles) == 1:
      self["index"] = ndxFiles[0]
      self.log.info( " using index file: " + self["index"] )
    topFiles = glob( path + "/*.tpr" )
    if len(topFiles) == 1:
      self["topology"] = topFiles[0]
      self.log.info( " using topology file: " + self["topology"] )
    # search for PCA blobs
    self["PCA_BLOBS"] = glob( path + "/*_pca.blob" )
    if self["PCA_BLOBS"] != []:
      for filename in self["PCA_BLOBS"]:
        self.log.info( " PCA already available from: " + filename )
    # search for DIH blobs
    dihBlobs = glob( path + "/*_dih.blob")
    if dihBlobs != []:
      self["DIH_BLOB"] = dihBlobs[0]
      self.log.info( " dihedrals already available from: " + dihBlobs[0] )
      if len(dihBlobs) > 1:
        self.log.warning( " only preloading of one dihedral BLOB supported right now" )
    # search for RGyr blobs
    self["RGYR_BLOBS"] = glob( path + "/*_rgyr.blob" )
    if self["RGYR_BLOBS"] != []:
      for filename in self["RGYR_BLOBS"]:
        self.log.info( " R_gyr already available from: " + filename )
    # search for RMSD blobs
    self["RMSD_BLOBS"] = glob( path + "/*_rmsd.blob" )
    if self["RMSD_BLOBS"] != []:
      for filename in self["RMSD_BLOBS"]:
        self.log.info( " RMSD values already available from: " + filename )
    return self


###################
# structural data

class Group:
  """
  represents group of molecule defined by index-file

  **attributes**

    - name:  name of the group (string)
    - atoms: atom numbers of the atoms biulding the group
  """
  def __init__(self, name=None, atoms=[]):
    self.name = name
    self.atoms = atoms

  def __str__(self):
    """print representation in index-file format"""
    buf = "[ " + self.name + " ]\n"
    for line in splitList(self.atoms, 15):
      buf += "  " + "  ".join( line ) + "\n"
    return buf

class Atom:
  """
  simple representation of an atom inside a molecule (peptide, protein, etc)

  **attributes**

    - atom:       the atom name
    - r:          coordinates of the atom
    - v:          velocity of the atom
    - residue:    the residue an atom is assigned to
  """
  def __init__(self, atom, r, v=None, residue=None):
    #TODO: self.atom -> self.name
    self.atom    = atom
    self.r       = r
    self.v       = v
    self.residue = residue

class Dihedrals:
  """
  list of dihedrals (phi/psi) of a trajectory

  **attributes**

    - dihedrals:  numpy.matrix of phi/psi coordinates.
                  cols define variable (phi1, psi1, phi2, psi2, ... )
                  rows define timesteps
  """
  def __init__(self, params, autosave=True):
    """
    params: dictionary with parameters (needed for g_rama)

    **needed parameters**

        - dih_input: trajectory filename for input
        - topology:  topology file 
        - DIH_BLOB:  [OPTIONAL] automatically load/save object
                     to filename given in the parameter
    """
    #TODO: move 'autosave' to params
    self.log = logging.getLogger( "Dihedrals" )
    self.dihedrals = None
    self.params = params
    self.autosave = autosave
    if "dih_input" in self.params.keys():
      if ("DIH_BLOB" in self.params.keys()) and self.autosave:
        # check, if given dih_input  matches the saved BLOB
        if os.path.normpath(self.params["DIH_BLOB"]) == os.path.normpath( self.blobFilename(self.params["dih_input"]) ):
          self.dihedrals = self.load()

    if (self.dihedrals == None) and "dih_input" in self.params.keys():
      # read if above checks for BLOB-file failed
      self.read()

  @staticmethod
  def blobFilename(inputFilename):
    """
    generate filename for a Dihedral-BLOB based on an input filename

    filename will be '<INPUT>_dih.blob', where
    <INPUT> is the filename of the input without '.gro' suffix.
    """
    return os.path.splitext(inputFilename)[0] + "_dih.blob"

  def save(self):
    """
    save data to a loadable BLOB

    filename will be '<INPUT>_dih.blob', where
    <INPUT> is the filename of the input without '.gro' suffix.
    """
    if not "dih_input" in self.params.keys():
      blobDump( self.dihedrals, self.blobFilename("dihedrals") )
    else:
      blobDump( self.dihedrals, self.blobFilename(self.params["dih_input"]) )

  def load(self):
    """load data from BLOB-file"""
    return blobLoad( self.params["DIH_BLOB"] )

  def fromXVG(self, filenamePhi, filenamePsi):
    """
    read dihedrals from XVG-files generated by g_angle.
    one file per angle-definition (phi, psi).
    """
    fhPhi = open(filenamePhi, 'r')
    fhPsi = open(filenamePsi, 'r')
    try:
      m = []
      while True:
        try:
          nextPhi = fhPhi.next()
          nextPsi = fhPsi.next()
          if not (nextPhi[0] == "#" or nextPhi[0] == "@"):
            # parse line
            m.append( [] )
            nextPhi = map(float, nextPhi.strip().split() )[2:]
            nextPsi = map(float, nextPsi.strip().split() )[2:]
            for i in range( len(nextPhi) ):
              m[-1].append(nextPhi[i])
              m[-1].append(nextPsi[i])
        except StopIteration:
          break
    finally:
      fhPhi.close()
      fhPsi.close()
      self.dihedrals = numpy.matrix( m )

    if self.autosave:
      self.save()
    return self

  
  def read(self, forceRead=False):
    """
    read dihedrals from .gro-file (given as 'dih_input' in params)
    """
    if (self.dihedrals != None) and (not forceRead):
      # abort, if dihedrals are already loaded
      return self
    # create unique tempfile for g_rama-output
    _, filename = tempfile.mkstemp(suffix=".xvg", prefix="dih_", dir=".")
    # delete the tempfile, just keep its name
    os.unlink( filename )
    # run g_rama to generate dihedrals
    cmd = "g_rama -f %s -s %s -o %s" % (self.params["dih_input"], self.params["topology"], filename)

    print cmd

    self.log.info("running 'g_rama' binary to generate dihedrals")
    gRamaProcess = pexpect.spawn( cmd )
    expId = gRamaProcess.expect( ["Can not open file", "File input/output error", pexpect.EOF], timeout=None )
    if expId == 0:
      raise "cannot find topology file '%s'" % self.params["topology"]
    elif expId == 1:
      raise "cannot find trajectory file '%s'" % self.params["dih_input"]
    elif 2:
      gRamaProcess.close()
    self.log.info("... 'g_rama' finished.")
    # read dihedrals from temporary file
    fh = open( filename, "r" )
    buf = {
      "firstGroup": None,
      "dih":        []
    }
    ###
    def perLine( line, ref ):
      line = line.strip()
      if not (line[0] == "@" or line[0] == "#"):
        line = line.split()
        phi = float(line[0])
        psi = float(line[1])
        group = line[2]
        if ref["firstGroup"] == None:
          ref["firstGroup"] = group
        # new line for next structure
        if ref["firstGroup"] == group:
          ref["dih"].append( [] )
        # append phi and psi to last structure in list
        ref["dih"][-1].append( phi )
        ref["dih"][-1].append( psi )
    ###
    # parse file and close
    linewise( fh, perLine, buf )
    # remove tempfile
    os.unlink( filename )
    self.dihedrals = numpy.matrix( buf["dih"] )
    # if autosave enabled, save data to file
    if self.autosave:
      self.save()
    # return reference to itself for method-concatenation
    return self

  def plot1DHist(self, dih=None, nGroup=None, bins=200, prep=None):
    """
    plot 1d-histogram of projections of a PC

    **arguments**

      - dih:    dihedral angle, either 'phi' or 'psi' (given as string)
      - nGroup: group of dihedral angle (start counting at 0)
      - bins:   # of bins
      - prep:   if set, diagram will be saved to a file
                called [prep]_[pc]_.eps.
                if not set, diagram will show up interactively.
    """
    plotAll = False
    if dih == 'phi':
      offset = 0
    elif dih == 'psi':
      offset = 1
    elif not dih and not nGroup:
      plotAll = True
    else:
      raise "unknown angle: " + dih

    def singlePlot(transp, nGroup, offset):
      labels = {0: 'phi', 1: 'psi'}
      dihDat = numpy.array( transp[2*nGroup + offset] ).flatten()
      h, _ = numpy.histogram( dihDat, bins=bins, range=(-180,180) )
      xRange = binning( -180, 180, bins )
      plt.plot( xRange, h.T, label="%s%i" % (labels[offset], nGroup+1) ) # +1, because first group is actually residue 2

    transp = self.dihedrals.transpose()
    if plotAll:
      # if dih / nGroup not given: plot all
      nGroups, _ = transp.shape
      nGroups /= 2
      for nGroup in range(nGroups):
        singlePlot(transp, nGroup, 0)
        singlePlot(transp, nGroup, 1)
    else:
      singlePlot(transp, nGroup, offset)

    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., prop=fontP)
    plt.ylabel( "population" )
    plt.xlabel( "[deg]" )

    plt.show()
    return self


  def filteredDihData(self, selectedIds):
    """
    generate dictionary with dihedral data for selected ids

    returns { "phi": { dihIndex1: [...], dihIndex2: [...], ... },
              "psi": { dihIndex1: [...], dihIndex2: [...], ... } }
    """
    # generate transpose of dihedral matrix
    transp = self.dihedrals.transpose()
    # helper function to get the correct dihedrals
    def filterIndices(x,y):
      buf = []
      for i in x:
        buf.append( y[i] )
      return buf
    result = { "phi": {}, "psi": {} }
    for dihIndex in self.params["plot_dihedrals"]:
      # filter unneeded phi/psi angles from list
      result["phi"][dihIndex] = filterIndices(
                                  selectedIds,
                                  numpy.array(transp[ 2*(dihIndex-1) ]).flatten()
                                )
      result["psi"][dihIndex] = filterIndices(
                                  selectedIds,
                                  numpy.array(transp[ 2*(dihIndex-1) +1 ]).flatten()
                                )
    return result

  def ramachandran(self, nGroup, bins=200):
    """
    create ramachandran plot

    **arguments**

      - nGroup:    number of phi/psi pair to be used (starting with 1)
    """
    transp = self.dihedrals.transpose()
    phi = numpy.array( transp[ (nGroup-1)*2    ] ).flatten() # -1, since first group (group 1) is at index 0
    psi = numpy.array( transp[ (nGroup-1)*2 +1 ] ).flatten()

    def replot( first=True ):
      plt.figure()
      if first:
        ax = plt.subplot( 111 )
      else:
        ax = plt.subplot( 211 )

      h, xedges, yedges = numpy.histogram2d( phi, psi, bins=bins )
      extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
      plt.imshow(h.T,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
      plt.colorbar()
      plt.xlabel( "$\\phi_{%i}$ [deg]" % (nGroup+1) ) # +1, since first group is actually phi/psi 2
      plt.ylabel( "$\\psi_{%i}$ [deg]" % (nGroup+1) )
      return ax

    ax = replot()

    #self.redonePlotsRama = 0
    def onselect(pos1, pos2):
      #ax = plt.subplot( 212 )
      fontP = FontProperties()
      fontP.set_size('x-small')
      self.params.setDefault( "plot_dihedrals", [nGroup] )
      # pos1 & pos2 are positions in 2D plane
      xMin = pos1.xdata
      xMax = pos2.xdata
      yMin = pos1.ydata
      yMax = pos2.ydata

      ax = replot( first=False )
      ax.add_patch( plt.Rectangle(
                      (xMin,yMin),
                       xMax-xMin,
                       yMax-yMin,
                       fill=False,
                       color='red',
                       linewidth=2.0
                    )
      )

      plt.subplot( 212 )

      # order values
      xMin, xMax = minMax( xMin, xMax )
      yMin, yMax = minMax( yMin, yMax )
      selectedIds = []
      # filter ids
      for i in range( phi.shape[0] ):
        inPhiRange = (xMin <= phi[i]) and (phi[i] <= xMax)
        inPsiRange = (yMin <= psi[i]) and (psi[i] <= yMax)
        if inPhiRange and inPsiRange:
          selectedIds.append( i )
      # plot dihedral histograms
      dihData = self.filteredDihData( selectedIds )
      xRange = binning( -180, 180, bins )
      for dihIndex in self.params["plot_dihedrals"]:
        #self.redonePlotsRama += 1
        h, _ = numpy.histogram( dihData["phi"][dihIndex], bins=bins, range=(-180,180) )
        plt.plot( xRange, h.T, label="phi%i" % (dihIndex+1) ) # +1, because first group is actually residue 2
        h, _ = numpy.histogram( dihData["psi"][dihIndex], bins=bins, range=(-180,180) )
        plt.plot( xRange, h.T, label="psi%i" % (dihIndex+1) )
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., prop=fontP)

      plt.ylabel( "population" )
      plt.xlabel( "[deg]" )
      # plot selection
      plt.show()

    span = matplotlib.widgets.RectangleSelector( ax, onselect )
    plt.show()


  def sinCosTransform(self):
    """
    transform dihedrals to sine/cosine representation for dPCA

    **output format**
.. math::
  [ [ \sin \psi, \cos \psi, \sin \phi, \cos \phi ], \dots ]
    """
    # some helper functions
    def setEvenDiags(m, val):
      """set even elements of diagonal to given value"""
      m.flat[1::m.shape[1]+2] = val
    def setOddDiags(m, val):
      """set odd elements of diagonal to given value"""
      m.flat[::m.shape[1]+2] = val
    def swapCols( m ):
      """return matrix with swapped even/odd columns"""
      r,c = m.shape
      if not c%2 == 0:
        raise Exception("cannot swap columns of matrix with odd number of columns")
      oddCols = m[ numpy.ix_( range(r), range(0,c,2) ) ]
      evenCols = m[ numpy.ix_( range(r), range(1,c,2) ) ]
      oddI = numpy.zeros( (c/2, c) )
      setOddDiags(oddI, 1)
      evenI = numpy.zeros( (c/2, c) )
      setEvenDiags(evenI, 1)
      return (oddCols*evenI) + (evenCols*oddI)

    # return new matrix with entries transformed to cosines/sines: m_new = cosTrans(m)
    cosTrans = numpy.vectorize( lambda x: numpy.math.cos(x) )
    sinTrans = numpy.vectorize( lambda x: numpy.math.sin(x) )
    cosines = cosTrans( self.dihedrals )
    sines = sinTrans( self.dihedrals )
    # swap phi1,psi1, ... to psi1,phi1, ...
    cosines = swapCols( cosines )
    sines = swapCols( sines )
    # intermix sines and cosines
    nCols = cosines.shape[1]
     # create matrices with every second (even or odd) diagonal entry equal to one
     # we use these to 'blow up' the cosine and sine matrices, setting every second
     # (even or odd) column in the 'blown up' matrices to zero. afterwards, the matrices
     # are easily combined by summation.
    m1 = numpy.matrix( numpy.zeros((nCols, 2*nCols)) )
    setOddDiags( m1, 1 )
    sines = sines * m1
    m2 = numpy.matrix( numpy.zeros((nCols, 2*nCols)) )
    setEvenDiags( m2, 1 )
    cosines = cosines * m2
    return sines + cosines
    


class InternalCoordinates:
  """represent/convert internal coordinates (distances, angles, dihedrals)"""
  def __init__(self, params=None):
    if params:
      self.params = params
    else:
      self.params = Params()
    self.bonds     = []
    self.angles    = []
    self.dihedrals = []

  def fromCartesians(self, cartCoords=None):
    if not cartCoords == None:
      cCoords = cartCoords
    elif 'input' in self.params:
      cCoords = GroFile( self.params ).readCoords()
    else:
      raise Exception( 
        "unable to convert cartesian coordinates to internals: no cartesians given and no input specified in 'params'"
      )
    cCoords = scipy.array(cCoords)
    nRows, nAtoms = cCoords.shape
    # we assume 3D space, therefore the number of atoms is
    # 1/3 of cartesian coords (x1,y1,z1,x2,y2,z2,...xN,yN,zN)
    nAtoms /= 3
    coordsN = lambda n,i: numpy.array( [cCoords[n][3*i], cCoords[n][3*i+1], cCoords[n][3*i+2]] )
    self.bonds = []
    self.angles = []
    self.dihedrals = []
    for n in range( nRows ):
      bondsCurStruct = []
      anglesCurStruct = []
      dihedralsCurStruct = []
      for i in range( nAtoms-1 ):
        a = coordsN(n,i)
        b = coordsN(n,i+1)
        # calculate bond lengths of consecutive atoms
        bondsCurStruct.append( numpy.sqrt(numpy.dot(a,b)) )
        if i < nAtoms-2:
          # calculate angles between atoms
          c = coordsN(n,i+2)
          anglesCurStruct.append( numpy.math.acos(numpy.dot((a-b),(c-b))) )
        if i < nAtoms-3:
          # calculate dihedrals angles
          d = coordsN(n,i+3)
          b1 = b-a
          b2 = c-b
          b3 = d-c
          dihedralsCurStruct.append(
            numpy.math.atan2(
              numpy.dot(numpy.math.sqrt(numpy.dot(b2,b2))*b1, numpy.cross(b2,b3)),
              numpy.dot(numpy.cross(b1,b2),numpy.cross(b2,b3))
            )
          )
      self.bonds.append( bondsCurStruct )
      self.angles.append( anglesCurStruct )
      self.dihedrals.append( dihedralsCurStruct )
      
  def toCartesians(self):
    #TODO: implement internal -> cartesian trafo
    pass


class Structure:
  """
  simple representation of a molecule

  **attributes**

    - atoms:    list of atoms ('Atom' objects)
    - cformat:  format of coordinates (DEFAULT: 'cartesian')
  """
  def __init__(self, atoms=None, cformat='cartesian'):
    if atoms:
      self.atoms = atoms
    else:
      self.atoms = []
    self.cformat = cformat

  def totalMass(self):
    masses = []
    for atom in self.atoms:
      masses.append( atomMasses[atom.atom] )
    return sum( masses )

  def centerOfMass(self):
    """
    returns center of mass as three dimensional vector (numpy.array)
    (attention: works only with atom definitions in cartesian coordinates)
    """
    com = scipy.zeros(3)
    for atom in self.atoms:
      com += scipy.array(atom.r) * atomMasses[atom.atom]
    return com / self.totalMass()

  def radiusOfGyration(self):
    """calculate radius of gyration of structure"""
    Rg = 0
    com = self.centerOfMass()
    for atom in self.atoms:
      diff = scipy.array(atom.r) - com
      Rg += scipy.dot(diff, diff) * atomMasses[atom.atom]
    return scipy.sqrt( Rg/self.totalMass() )

  def __str__(self):
    """prints structure in simple 'x1 y1 z1 x2 y2 z2 ...' representation"""
    buf = []
    for atom in self.atoms:
      buf.extend( [atom.r[0], atom.r[1], atom.r[2]] )
    return " ".join( map(str,buf) )

  def loadFromXYZ(self, xyzArray, ref):
    """load structure from xyz-array (scipy-array);
        ref is a Structure object which defines a reference for atom names and residues"""
    xyzArray = xyzArray.flatten().tolist()
    nAtoms = len(xyzArray) / 3
    self.atoms = []
    for i in range(nAtoms):
      atom = copy.copy( ref.atoms[i] )
      atom.r = [ xyzArray[3*i], xyzArray[3*i+1], xyzArray[3*i+2] ]
      atom.v = scipy.zeros(3).tolist()
      self.atoms.append( atom )

  def translate(self, d):
    """translate structure in given direction"""
    for i in range( len(self.atoms) ):
      self.atoms[i].r += d
    

class Trajectory:
  """
  represents an MD trajectory (arbitrary coordinates)

  **attributes**

    - dt:          timestep of trajectory [ns]
    - t:           length of trajectory [ns]
    - structures:  list of structures
  """
  def __init__(self, dt=0.0, t=0.0):
    self.dt = dt
    self.t  = t
    self.structures = []

  def filter( ids ):
    """keep only structures whose id is in the given list of ids"""
    self.structures = map(  lambda i,s: s,
                            filter( lambda i,s: i in ids,  enumerate(self.structures) )
    )

  def translateCOM(self):
    """translate structures to set center of mass to origin"""
    s = self.structures
    for i in range( len(s) ):
      s[i].translate( (-1)* s[i].centerOfMass() )

  def extractCoords(self):
    """extract coords from structures and return them as numpy-matrix
      (columns: coords, rows: structs)"""
    m = []
    for s in self.structures:
      # join all atomic coordinates of one structure
      # in one list and append this to the matrix
      atomCoords = map( lambda a: a.r, s.atoms )
      m.append( reduce(lambda x,y: x+y, atomCoords) )
    return numpy.matrix( m )



###############
# binary interfaces

class G_rms:
  @staticmethod
  def blobFilename( inputFilename ):
    """
    generate filename for a RMSD-BLOB based on an input filename and used group.

    filename of BLOB will be '<INPUT>_<GROUP>_rmsd.blob', where
    <INPUT> is the filename of the input without '.gro' suffix
    and group is the group of atoms used for computing the rmsd.
    """
    return os.path.splitext(inputFilename)[0] + "_rmsd.blob"

  def __init__(self, params):
    self.log = logging.getLogger("g_rms")
    self.params = params
    self.params.setDefault( "project_dir", "." )
#    if "bmd" in platform.node():
#      # assume local machine
#      self.params.setDefault( "gromacs_binaries", "/usr/bin/" )
#    else:
#      # assume BWGRID
#      self.params.setDefault( "gromacs_binaries", "/opt/bwgrid/chem/gromacs/4.5.5-openmpi-1.4.3-intel-12.0/bin/" )
    self.params.setDefault( "blob_autoload", True )
    self.RMSD = []
    # load from BLOB (if available)
    self.blob = self.blobFilename( self.params["input"] )
    for filename in self.params["RMSD_BLOBS"]:
      if ( os.path.basename(filename) == self.blob ) and self.params["blob_autoload"]:
        self.RMSD = blobLoad( self.blob )
        break
    else:
      self.run()
  
  def run(self):
    # create unique tempfile for g_rms-output
    _, temp = tempfile.mkstemp(suffix=".xvg", prefix="rmsd_", dir=".")
    # delete the tempfile, just keep its name
    os.unlink( temp )
    # create unique tempfile for trjconv-output
    _, tempRef = tempfile.mkstemp(suffix=".gro", prefix="rmsd_reduced_ref_", dir=".")
    # delete the tempfile, just keep its name
    os.unlink( tempRef )
    # use trjconv to convert reference structure to reduced dataset
    tcParams = Params()
    tcParams.load()
    tcParams["input"] = self.params["reference"]
    tcParams["groupOut"] = self.params["groupRMSD"]
    tcParams["reference"] = self.params["reference"]
    tcParams["output"] = tempRef
    Trjconv( tcParams ).reduceToGroup()

    # prepare command and parameters for g_rms
    cmd = "%s/g_rms -fit none -f %s -s %s -o %s" % (
              self.params["gromacs_binaries"],
              self.params["input"],
              tempRef,
              temp
    )

    ### run g_rms
    # save original directory
    origDir = os.getcwd()
    # go to project dir
    os.chdir( self.params["project_dir"] )
    self.log.info( "running '%s' with group '%s'",
                    cmd, self.params["groupRMSD"]
    )
    childProcess = pexpect.spawn( cmd )
    # set group
    childProcess.expect( "Select a group" )
    childProcess.sendline( "0" )
    # close process
    childProcess.expect( pexpect.EOF, timeout=None )
    childProcess.close()
    self.log.info( "...finished 'g_rms' with group '%s'.", self.params["groupRMSD"] )
    # go back to original directory
    os.chdir( origDir )

    ### parse .xvg file
    def lineParser( line, outRef ):
      line = line.strip()
      if line == "" or line[0] == "#" or line[0] == "@":
        return
      else:
        # sec. col. is RMSD
        outRef.append( float(line.split()[1]) )

    fh = open( temp, 'r' )
    RMSD = []
    linewise( fh, lineParser, RMSD )
    # delete the tempfiles
    os.unlink( temp )
    os.unlink( tempRef )
    
    self.RMSD = scipy.array( RMSD )
    if self.params["blob_autoload"]:
      # save data
      blobDump( self.RMSD, self.blob )

    return self.RMSD


class G_rmsf:
  @staticmethod
  def blobFilename( inputFilename, groupName ):
    """
    generate filename for a RMSF-BLOB based on an input filename and used group.

    filename of BLOB will be '<INPUT>_<GROUP>_rmsf.blob', where
    <INPUT> is the filename of the input-file without suffix (e.g. '.gro')
    and group is the group of atoms used for computing the RMSF.
    """
    return os.path.splitext(inputFilename)[0] + "_" + groupName + "_rmsf.blob"

  def __init__(self, params):
    self.log = logging.getLogger("g_rmsf")
    self.params = params
    self.params.setDefault("project_dir", ".")
    self.params.setDefault("blob_autoload", True)
    self.params.setDefault("group", "C-alpha")
    self.blob = self.blobFilename(self.params["input"], self.params["group"])
    for filename in self.params["RMSF_BLOBS"]:
      if (os.path.basename(filename) == self.blob) and self.params["blob_autoload"]:
        self.rmsf = blobLoad(self.blob)
        break
    else:
      self.run()

  def run(self):
    #TODO: finish
    pass




class G_gyrate:
  @staticmethod
  def blobFilename( inputFilename, groupName ):
    """
    generate filename for a RGyr-BLOB based on an input filename and used group.

    filename of BLOB will be '<INPUT>_<GROUP>_rgyr.blob', where
    <INPUT> is the filename of the input without '.gro' suffix
    and group is the group of atoms used for computing the radius of gyr.
    """
    return os.path.splitext(inputFilename)[0] + "_" + groupName + "_rgyr.blob"

  def __init__(self, params):
    # create logger
    self.log = logging.getLogger("g_gyrate")
    self.params = params
    self.params.setDefault( "project_dir", "." )
#    if "bmd" in platform.node():
#      # assume local machine
#      self.params.setDefault( "gromacs_binaries", "/usr/bin/" )
#    else:
#      # assume BWGRID
#      self.params.setDefault( "gromacs_binaries", "/opt/bwgrid/chem/gromacs/4.5.5-openmpi-1.4.3-intel-12.0/bin/" )
    self.params.setDefault( "blob_autoload", True )
    self.params.setDefault( "groupRgyr", "System" )
    self.Rgyr = []
    # load from BLOB (if available)
    self.blob = self.blobFilename( self.params["input"], self.params["groupRgyr"] )
    for filename in self.params["RGYR_BLOBS"]:
      if ( os.path.basename(filename) == self.blob ) and self.params["blob_autoload"]:
        self.Rgyr = blobLoad( self.blob )
        break
    else:
      self.run()

  def run(self):
    # create unique tempfile for g_gyrate-output
    _, temp = tempfile.mkstemp(suffix=".xvg", prefix="rGyr_", dir=".")
    # delete the tempfile, just keep its name
    os.unlink( temp )
    # prepare command and parameters
    cmd = "%sg_gyrate -f %s -s %s -n %s -o %s" % (
              self.params["gromacs_binaries"],
              self.params["input"],
              self.params["reference"],
              self.params["index"], temp
    )

    ### run g_gyrate
    # save original directory
    origDir = os.getcwd()
    # go to project dir
    os.chdir( self.params["project_dir"] )
    self.log.info( "running '%s' with group '%s'",
                    cmd, self.params["groupRgyr"]
    )
    childProcess = pexpect.spawn( cmd )
    # set group
    childProcess.expect( "Select a group" )
    childProcess.sendline( self.params["groupRgyr"] )
    # close process
    childProcess.expect( pexpect.EOF, timeout=None )
    childProcess.close()
    self.log.info( "...finished 'g_gyrate' with group '%s'.", self.params["groupRgyr"] )
    # go back to original directory
    os.chdir( origDir )

    ### parse .xvg file
    def lineParser( line, outRef ):
      line = line.strip()
      if line == "" or line[0] == "#" or line[0] == "@":
        return
      else:
        # sec. col. is Rgyr (total)
        outRef.append( float(line.split()[1]) )

    fh = open( temp, 'r' )
    Rgyr = []
    linewise( fh, lineParser, Rgyr )
    # delete the tempfile
    os.unlink( temp )
    
    self.Rgyr = scipy.array( Rgyr )
    if self.params["blob_autoload"]:
      # save data
      blobDump( self.Rgyr, self.blob )


class PairwiseBinary:
  """generic caller for pairwise interaction"""
  def __init__(self, params=Params()):
    self.log = logging.getLogger("generic pairwise interaction")
    self.params = params
    self.params.setDefault( "project_dir", "." )
#    if "bmd" in platform.node():
#      # assume local machine
#      self.params.setDefault( "gromacs_binaries", "/usr/bin/" )
#    else:
#      # assume BWGRID
#      self.params.setDefault( "gromacs_binaries", "/opt/bwgrid/chem/gromacs/4.5.5-openmpi-1.4.3-intel-12.0/bin/" )
    # call localInit of child-class (if implemented)
    self.localInit()

  def localInit(self):
    """implement this in child-classes to use parent-constructor with defaults"""
    pass

  def pairwiseInteraction(self, cmd, groups, cmdOpts, colDistribution=1, colRawdata=1):
    """generate scipy array with data from pairwise interaction between groups given as list of groupnames"""
    nGroups = len(groups)
    rMatrices = None
    dMatrices = None
    for i in range(1,nGroups):
      for j in range(i):
        # create unique tempfiles for g_mindist-output
        _, tempRawdata      = tempfile.mkstemp(suffix=".xvg", prefix="rawdata_", dir=".")
        _, tempDistribution = tempfile.mkstemp(suffix=".xvg", prefix="distribution_", dir=".")
        # delete the tempfile, just keep its name
        os.unlink( tempRawdata      )
        os.unlink( tempDistribution )
        # prepare command
        runCmd = cmd
        if "rawdata" in cmdOpts.keys() and cmdOpts["rawdata"] != None:
          runCmd += " " + cmdOpts["rawdata"] + " " + tempRawdata
        if "distribution" in cmdOpts.keys() and cmdOpts["distribution"] != None:
          runCmd += " " + cmdOpts["distribution"] + " " + tempDistribution
        # run binary
        self.log.info( "running '%s' for groups '%s' and '%s'" % (runCmd, groups[i], groups[j]))
        childProcess = pexpect.spawn( runCmd )
        childProcess.expect( "Select a group" )
        childProcess.sendline( groups[i] )
        childProcess.expect( "Select a group" )
        childProcess.sendline( groups[j] )
        # close process
        childProcess.expect( pexpect.EOF, timeout=None )
        childProcess.close()
        if "distribution" in cmdOpts.keys() and cmdOpts["distribution"] != None:
          # parse output file and store to sparse arrays
          nInteractions = []
          try:
            fh = open(tempDistribution, 'r')
            for line in fh:
              line=line.strip()
              if line    == "":  continue
              if line[0] == "#": continue
              if line[0] == "@": continue
              nInteractions.append( int(line.split()[colDistribution]) )
          finally:
            fh.close()
          if i==1 and j==0:
            # first run: setup list of matrices
            dMatrices = map( lambda x: scipy.sparse.dok_matrix( scipy.zeros( (nGroups,nGroups) ) ),
                             range(len(nInteractions))
                        )
          for k in range( len(nInteractions) ):
            n = dMatrices[k]
            # need this because of bug in scipy.sparse.dok_matrix:
            # assigning zero to already zero field fails with exception
            if not (n[i,j] == 0 and nInteractions[k] == 0):
              n[i,j] = nInteractions[k]
        if "rawdata" in cmdOpts.keys() and cmdOpts["rawdata"] != None:
          rawdata = []
          try:
            fh = open(tempRawdata, 'r')
            for line in fh:
              line = line.strip()
              if line    == "":  continue
              if line[0] == "#": continue
              if line[0] == "@": continue
              rawdata.append( float(line.split()[colRawdata]) )
          finally:
            fh.close()

          if i==1 and j==0:
            rMatrices = map( lambda x: scipy.sparse.dok_matrix( scipy.zeros( (nGroups,nGroups) ) ),
                             range(len(rawdata))
                        )
          for k in range( len(rawdata) ):
            r = rMatrices[k]
            # need this because of bug in scipy.sparse.dok_matrix:
            # assigning zero to already zero field fails with exception
            if not (r[i,j] == 0 and rawdata[k] == 0):
              r[i,j] = rawdata[k]

        # delete tempfiles
        try:
          os.unlink( tempRawdata )
        except OSError: pass
        try:
          os.unlink( tempDistribution )
        except OSError: pass

    return (rMatrices, dMatrices) # lists of matrices with rawdata and distribution



class G_sgangle(PairwiseBinary):

  def localInit(self):
    self.log = logging.getLogger("g_sgangle")

  def angleMatrices(self, groups):
    """compute list of angle matrices"""
    # prepare command and parameters
    cmd = "%sg_sgangle -f %s -s %s -n %s" % (
              self.params["gromacs_binaries"],
              self.params["trajectory"],
              self.params["reference"],
              self.params["index"]
    )
    cmdOpts = {}
    # not interested in the raw data
    cmdOpts["rawdata"] = "-oa"
    # ... but in the distribution (i.e. number of contacts)
    cmdOpts["distribution"] = None
    return self.pairwiseInteraction(cmd, groups, cmdOpts, colRawdata=2)[0]

  def angleClustering(self, aMs, angleCutoffs, cMs=None):
    """take angle pairs from angle matrices and cluster them into
       the intervals defined by the 'angleCutoffs'.
       e.g.: angleCutoffs = [120,240] will sort a given angle
       into the ranges [0,120), [120,240) and [240,180].
       40 deg will in this example be matched to 1 (index 0 == no contact)
       167 deg -> 2, 301 deg -> 3, etc.

       returns list of matrices with cluster assignments.
       if cMs (contact matrices) are given, check only those angles that
       are in contact.
    """
    nMolecules, _ = aMs[0].shape
    nFrames = len(aMs)
    clusts = map( lambda x: scipy.sparse.dok_matrix( scipy.zeros( (nMolecules,nMolecules) ) ),
                  range(nFrames)
             )
    for i in range(nFrames):
      aM = aMs[i]
      for r in range(nMolecules):
        for c in range(r, nMolecules):
          if (r == c) or (cMs and cMs[i][r,c] == 0 and cMs[i][c,r] == 0):
            # use contact information + no contact for this pair: skip it!
            continue
          a = aMs[i][r,c]
          if a == 0:
            a = aMs[i][c,r]
          for cut, cutoff in enumerate(angleCutoffs):
            if a < cutoff:
              clusts[i][r,c] = cut+1
              break
          else:
            clusts[i][r,c] = len(angleCutoffs)+1
    return clusts


class G_mindist(PairwiseBinary):

  def localInit(self):
    self.log = logging.getLogger("g_mindist")
    self.params.setDefault( "contact_dist", 0.45 )
    
  def contactMatrices(self, groups):
    """compute list of contact matrices"""
    # prepare command and parameters
    cmd = "%sg_mindist -f %s -s %s -n %s -d %s" % (
              self.params["gromacs_binaries"],
              self.params["trajectory"],
              self.params["reference"],
              self.params["index"],
              str(self.params["contact_dist"])
    )
    cmdOpts = {}
    # not interested in the raw data
    cmdOpts["rawdata"] = None
    # ... but in the distribution (i.e. number of contacts)
    cmdOpts["distribution"] = "-on"
    # delete default mindist-file
    try:
      os.unlink("mindist.xvg")
    except OSError: pass
    return self.pairwiseInteraction(cmd, groups, cmdOpts)[1]

  def aggregates(self, groups=None, cMs=None):
    """return list with aggregates for every timestep.
       either define 'groups' to compute contacts between defined monomers or give list
       with a contact matrix for every timestep (cMs)."""
    if not groups and not cMs:
      raise Exception("need to specify either list of groups or list of contact matrices")
    if groups:
      # groups given: compute contact matrices
      cMs = self.contactMatrices(groups)

    def mergeAggs( aggs ):
      """tests all aggregates for mergeability. if two aggregates are mergeable,
         they will be merged (inline) and the new list of aggregates will be returned"""
      reduceOr  = lambda l: reduce(lambda x,y: x or y, l)
      mergeable = lambda x,y: reduceOr([ n in aggs[x] for n in aggs[y] ])
      for i in range(len(aggs)):
        for j in range(i+1,len(aggs)):
          if mergeable(i,j):
            aggs[i] = list( set(aggs[i] + aggs[j]) )
            aggs.remove( aggs[j] )
            break
      return aggs

    # list of aggregates / timestep
    aggList = []
    nMonomers = cMs[0].shape[0]
    for a in range( len(cMs) ):
      aggregates = []
      for i in range(nMonomers):
        aggregates.append([i])
        for j in range(i,nMonomers):
          if cMs[a][j,i] != 0:
            aggregates[i].append(j)
      # merge aggregates that belong together
      oldLength = len(aggregates)+1
      while len(aggregates) != oldLength:
        oldLength = len(aggregates)
        aggregates = mergeAggs(aggregates)
      aggList.append( aggregates )
    # return list with aggregates for every timestep
    return aggList

  def transitionMatrixSummedRatios(self, aggregates, nMonomers):
    """
    build transition matrix of polymers
    schema: take aggregates of step i+1 and check, which aggregates of step i spent a monomer
            for the specified agg. at i+1.
            e.g.:     i                     i+1
                  [ [1,2], [3,4,5] ]  ->  [ [1,2,3,4], [5] ]
            first aggregate at i+1: [1,2,3,4] has monomers from [1,2] and [3,4,5].
            therefore it is made up to 2/4=1/2 from a dimer and 2/4=1/2 from a trimer.
            second aggregate is [5], made completely from a trimer ([3,4,5] in step i), giving a ratio of 1.
            resulting matrix:
           to   I  II  III   IV    V
       from
          I  [ [0,  0,  0,   0,    0],
         II    [0,  0,  0,   0.5,  0],
        III    [1,  0,  0,   0.5,  0],
         IV    [0,  0,  0,   0,    0],
          V    [0,  0,  0,   0,    0]  ]
            read matrix as:  'to' is made up by a ratio of '#' of 'from'.
            i.e.  "one tetramer is built 0.5 by monomers from a dimer and 0.5 by monomers from a trimer".
    """
    tM = scipy.zeros( (nMonomers,nMonomers) )
    for iStep in range( len(aggregates)-1 ):
      tMlocal = scipy.zeros( (nMonomers,nMonomers) )
      for agg in aggregates[iStep+1]:
        nFormer = lambda x,a: [len(a[j]) for j in range(len(a)) if x in a[j] ][0]
        for i in agg:
          nTo = len(agg)
          nFrom = nFormer(i,aggregates[iStep])
          tMlocal[nFrom-1, nTo-1] += 1.0/nTo
      tM += tMlocal
    return tM


  def transitionMatrixSingleMolecules(self, aggregates):
    """
    count every time a single molecule goes from one state (e.g. 'monomer') to
    another state (e.g. 'trimer').
    row-index defines from-state, col-index defines to-state.
    """
    a = aggregates
    # get number of monomers
    nMonomers = sum( [ len(g) for g in a[0] ] )
    tm = scipy.zeros( (nMonomers, nMonomers) )
    def whichPoly( g, i ):
      for s in g:
        if i in s:
          return len(s)
    for g in range(len(a)-1):
      #for i in range(1,nMonomers+1):
      for i in range(0,nMonomers):
        # polyFrom == type of polymer the current monomer i belongs to
        # e.g.: aggregates [ [1,2,3], [4,5] ] yield
        #       j=3 for i=1,2 or 3  and  j=2 for i=4 or 5
        polyFrom = whichPoly( a[g], i )
        polyTo   = whichPoly( a[g+1], i )
        tm[polyFrom-1][polyTo-1] += 1
    return tm


  def transitionMatrixCombinations(self, aggregates):
    """
    compute the transition matrix of the polymer combinations given by self.aggregateCombinations(...).
    """
    comb = self.aggregateCombinations( aggregates )
    orderedKeys = map( lambda x: x[0],   sorted(comb.iteritems(), key=operator.itemgetter(1)) )
    tm = scipy.zeros( (len(orderedKeys), len(orderedKeys)) )
    for i in range( len(aggregates)-1 ):
      keyFrom = str(sorted(aggregates[i  ]))
      keyTo   = str(sorted(aggregates[i+1]))
      iFrom = orderedKeys.index(keyFrom)
      iTo   = orderedKeys.index(keyTo)
      tm[iFrom,iTo] += 1
    return ( orderedKeys, tm )

  def aggregateCombinations(self, aggregates):
    """
    compute all combinations of polymers the aggregates form.
    a trajectory with three molecules can, for example, cluster from
    three monomers to a monomer and a dimer to a trimer.
    with molecule-ids 0,1,2 this would give the combinations
    [ [0], [1], [2] ],
    [ [0,1], [2]],
    [ [0,1,2] ],
    with 0 & 1 forming the dimer.
    """
    combinations = {}
    for c in aggregates:
      c = str(sorted(c))
      if c in combinations.keys():
        combinations[c] += 1
      else:
        combinations[c]  = 1
    return combinations


  def normalizeTransitionMatrix(self, tm):
    # normalize transition matrix over rows
    nRows, _ = tm.shape
    for i in range( nRows ):
      tm[i] = tm[i] / tm[i].sum()
    return tm

  def labelCheck(self, nStates, labels):
    if labels:
      if (len(labels)) != nStates:
        raise "# of labels does not mach # of states"
    else:
      labels = map(str, range(nStates))
    return labels
  
  def printTransitionMatrix(self, tm, labels=None, separator=";"):
    nStates, _ = tm.shape
    labels = self.labelCheck(nStates, labels)
    labels = map( lambda x: '"'+x+'"', labels )
    buf = separator + separator.join( labels ) + "\n"
    for i in range(nStates):
      buf += labels[i] + separator + separator.join( map(str, tm[i]) ) + "\n"
    print buf.strip()

  def showNetwork(self, tm, labels=None, widthScale=1, toSelf=True):
    """print graphical representation of transition matrix"""
    nStates, _ = tm.shape
    labels = self.labelCheck(nStates, labels)
    g = nx.DiGraph()
    for l in labels:
      g.add_node(l)
    # add edges
    for i in range( nStates ):
      for j in range( nStates ):
        if (i!=j) or toSelf:
          g.add_edge( labels[i], labels[j],  weight=tm[i,j] )
          g.add_edge( labels[j], labels[i],  weight=tm[j,i] )
    # define labels for plot
    weights = []
    edgeLabels = {}
    for u,v in g.edges():
      w = g.get_edge_data(u,v)['weight']
      weights.append( w * widthScale )
      if w >= 0.001:
        edgeLabels[ (u,v) ] = "%0.3f" % w
      else:
        edgeLabels[ (u,v) ] = ""
    # disable coordinate system in matplotlib
    ax = plt.axes(frameon=False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    # apply layout and plot
    #pos = nx.spring_layout(g, iterations=20)
    pos = nx.circular_layout(g)
    nx.draw_networkx_edges(g,pos,width=weights)
    nx.draw_networkx_nodes(g,pos)
    nx.draw_networkx_labels(g,pos)
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeLabels)
    plt.show()


class Trjconv:
  """wrapper around the trjconv binary"""
  def __init__(self, params, projectDirectory=".", pathToBinary="trjconv"):
    # TODO: move 'projectDirectory' and 'pathToBinary' to params
    self.params = params
    self.projectDir = projectDirectory
    self.binary = pathToBinary


  def reduceToGroup(self):
    """
    parameters:

      groupOut: name of the group as defined in index-file
    """
    log = logging.getLogger("trjconv")
    # save original directory
    origDir = os.getcwd()
    # go to project dir
    os.chdir( self.projectDir )
    # run single reference fit
    cmd  = self.binary + " -f " +   self.params["input"]
    cmd +=               " -o " +   self.params["output"]
    cmd +=               " -s " +   self.params["reference"]
    cmd +=               " -n " +   self.params["index"]
    cmd +=               " -fit none"
    # run binary
    log.info( "running '%s' to reduce structure to group '%s'" % (cmd, self.params["groupOut"]))
    childProcess = pexpect.spawn( cmd )
    # set output gr oup
    childProcess.expect( "Select a group" )
    childProcess.sendline( self.params["groupOut"] )
    # close process
    childProcess.expect( pexpect.EOF, timeout=None )
    childProcess.close()
    log.info( " ... finished reduction to '%s'.", self.params["groupOut"] )
    # go back to original directory
    os.chdir( origDir )


  def fit(self):
    """
    run fit on trajectory

    **parameters**

      - input:     trajectory filename for input
      - output:    trajectory filename for output
      - reference: file with reference structure
      - index:     index file
      - mode:      "rot+trans" (DEFAULT), "rotxy+transxy", "translation", "transxy" or "progressive"
      - groupFit:  name of group that should be fitted to
      - groupOut:  name of group that should be written out (DEFAULT: same as 'groupFit')

    **differences in modes**

      - rot+trans:     fit to one structure (usually first in trajectory) by translating molecule to
                       the center and rotating (in 3D).
      - rotxy+transxy: same as 'rot+trans', but only in 2D (x & y).
      - translation:   only translate to center (3D).
      - transxy:       same as 'translation', but only in 2D (x & y).
      - progressive:   essentially the same as 'rot+trans', but uses the last fitted structure as new
                       reference instead of fitting to the first structure.
    """
    log = logging.getLogger("trjconv")
    # set default parameters
    self.params.setDefault("mode", "rot+trans")
    self.params.setDefault("groupOut", self.params["groupFit"])
    # save original directory
    origDir = os.getcwd()
    # go to project dir
    os.chdir( self.projectDir )
    # run single reference fit
    cmd  = self.binary + " -f " +   self.params["input"]
    cmd +=               " -o " +   self.params["output"]
    cmd +=               " -s " +   self.params["reference"]
    cmd +=               " -n " +   self.params["index"]
    cmd +=               " -fit " + self.params["mode"]
    # run binary
    log.info( "running '%s' with fit-group '%s' and output-group '%s'",
              cmd, self.params["groupFit"], self.params["groupOut"]
    )
    childProcess = pexpect.spawn( cmd )
    # set fit group
    childProcess.expect( "Select group for least squares fit" )
    childProcess.sendline( self.params["groupFit"] )
    # set output group
    childProcess.expect( "Select group for output" )
    childProcess.sendline( self.params["groupOut"] )
    # close process
    childProcess.expect( pexpect.EOF, timeout=None )
    childProcess.close()
    log.info( "...finished fit via 'trjconv' on group '%s'.", self.params["groupFit"] )
    # go back to original directory
    os.chdir( origDir )



#################
# file type interfaces

class IndexFile:
  """reads, creates and alters GROMACS index files"""
  def __init__(self, filename):
    """represent index file with given filename. if file exists, it will be loaded automatically."""
    #TODO: move 'filename' to params
    self.indexFile = filename
    self.groups = {}
    if self.indexFile and os.path.isfile(self.indexFile):
      self.read()

  def generate(self, groFileName):
    """generate index groups from gro-file"""
    try:
      fh = open(groFileName, 'r')
      fh.next() # get rid of 'Generated ...' line
      fh.next() # get rid of line with number of atoms
      backbone = {}
      system = {}
      while True:
        line = fh.next().strip().split()
        if len(line) == 3:
          # next structure: end loop
          break
        atomNumber  = int(line[2])
        atomType    = line[1]
        if (atomType == "N") or (atomType == "CA") or (atomType == "C"):
          backbone[atomNumber] = atomType
        system[atomNumber] = atomType
    finally:
      fh.close()
    # create groups
    self.groups['System'] = system.keys()
    self.groups['Backbone'] = backbone.keys()
    print 'system: ', self.groups['System']
    print 'backbone: ', self.groups['Backbone']
    nSub = 0
    for i in range( 0, len(backbone.keys())-4, 3):
      nSub += 1
      #TODO: check if indices are correct in index file generation for PhiPsi groups
      self.groups['Psi%i' % nSub]    = backbone.keys()[i:i+4]
      self.groups['Phi%i' % nSub]    = backbone.keys()[i+1:i+5]
      self.groups['PsiPhi%i' % nSub] = backbone.keys()[i:i+5]

  
  def read(self):
    """read list of groups from index file and return them as dictionary with groupname as key"""
    groupName = None
    atomNumbers = None
    fh = open( self.indexFile, "r" )
    try:
      for line in fh.readlines():
        line = line.strip()
        if line != "":
          if line[0] == "[":
            if groupName:
              self.groups[ groupName ] = Group( groupName, atomNumbers.strip().split() )
            # read new name
            groupName = line[1:-1].strip()
            atomNumbers = ""
          else:
            # read atom numbers
            atomNumbers += " " + line
    finally:
      fh.close()
    return self.groups

  def write(self):
    """write group definitions to index file"""
    def group2string( groupName, groupDef ):
      buf = "[ " + groupName + " ]\n"
      buf += " ".join( map(str, groupDef) )
      buf += "\n"
      return buf
    try:
      fh = open(self.indexFile, 'w')
      for group in self.groups:
        fh.write( group2string(group, self.groups[group]) )
    finally:
      fh.close()


class GroFile:
  """represents GROMACS .gro trajectory files"""
  def __init__(self, params):
    self.params = params
    self.log = logging.getLogger( " GroFile " )

  def readCoords(self, ids=None):
    """
    read coords from given trajectory file and return as numpy.matrix
    if ids is a list of IDs, read only structures with ids given by list.

    **format**
|        x1 y1 z1 x2 y2 z2 ...
|        .
|        .
|        .
|        v   
|      timesteps
    """
    # TODO: change to 'linewise' parser
    self.log.info( " reading coords from trajectory: " + self.params["input"] )
    m = []
    fh = open( self.params["input"], "r" )
    curId = -1 
    try:
      while 1:
        try:
          line = fh.next()
          if 'Generated' in line:
            curId += 1
            # go on with next structure if filter ids are given
            # and this structure does not belong to them
            if ids:
              if curId not in ids:
                continue

            nAtoms = int( fh.next().strip() )
            buf = []
            for i in range(nAtoms):
              line = fh.next().strip().split()
              buf.extend( [ float(line[3]), float(line[4]), float(line[5]) ] )
            m.append( buf )
        except StopIteration:
          break
    finally:
      fh.close()
    self.log.info( " ... finished." )
    return numpy.matrix(m)

  def framewise(self, func, nFrames=None):
    """
    read file framewise, generate structure from frame and apply 'func( struct )'.
    abort after nFrames (default: None = read all).
    """
    # TODO: currently ignores box vectors
    # TODO: currently ignores t / dt
    frameCounter = 0
    fh = open( self.params["input"], "r" )
    try:
      while 1:
        try:
          line = fh.next()
          if 'Generated' in line:
            frameCounter += 1
            nAtoms = int( fh.next().strip() )
            struct = Structure()
            for i in range(nAtoms):
              line = fh.next().strip().split()
              # append atom to structure
              if len(line) == 6:
                line.extend( [0,0,0] )
              struct.atoms.append(
                Atom( atom    = line[1],
                      r       = scipy.array( map(float, [line[3], line[4], line[5]]) ),
                      v       = scipy.array( map(float, [line[6], line[7], line[8]]) ),
                      residue = line[0]
                )
              )
            # apply given function
            func( struct )
            if nFrames and (frameCounter >= nFrames):
              break # got enough frames: abort
        except StopIteration:
          break # exit the while-loop
    finally:
      fh.close()

  def read(self, nFrames=None):
    traj = Trajectory()
    self.framewise( traj.structures.append, nFrames )
    return traj

  def write(self, traj, fh=None, mode="w"):
    """
    write trajectory to .gro-file

    **arguments**

      - traj:   a Trajectory-object
      - mode:   'w' for overwrite (DEFAULT), 'a' for append
    """
    # TODO: currently ignores box vectors
    # TODO: currently ignores top title ( 'Protein in water' hardcoded )
    if fh:
      fileGiven = True
    else:
      fileGiven = False
      fh = open( self.params["output"], mode )
    templHead   = "Generated by PGT : Protein in water t= %9.5f\n"
    templNAtoms = "%5d\n"
    templStruct = "%8s%7s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n"
    templBoxVec = "%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n"
    try:
      t  = 0.0
      dt = traj.dt
      nAtoms = len(traj.structures[0].atoms)
      for i in range( len(traj.structures) ):
        fh.write( templHead % t )
        fh.write( templNAtoms % nAtoms )
        curStruct = traj.structures[i]
        for iAtom in range(1,nAtoms+1):
          atom = curStruct.atoms[iAtom-1]
          fh.write( templStruct % (atom.residue,atom.atom,iAtom,atom.r[0],atom.r[1],atom.r[2],atom.v[0],atom.v[1],atom.v[2]) )
        fh.write( templBoxVec % tuple(scipy.zeros(9)) )
        t += dt
    finally:
      if not fileGiven:
        fh.close()


######################
# statistics
class PCA:
  """
  perform a principal component analysis on the given matrix

  ** parameter **
    - PCA_proj: list with eigenvectors for eigenvector projection. None for all eigenvectors.
            e.g.: params['proj'] = [1,2,3]  -> project data only on first three eigenvectors

    *** mode ***
      perform PCA using either covariance- or correlation matrix.
      - PCA_mode='cov':  use covariance matrix (DEFAULT)
      - PCA_mode='corr': use correlation matrix
      - PCA_cols=[c1, c2, c3, ... ]:  only perform PCA on given columns (base-1)

  ** attributes **
    - projection: trajectory projected onto PCs
    - fracs:      fractions of variance for PCs
  """
  @staticmethod
  def blobFilename( params ):
    """
    generate filename for a PCA-BLOB based on an input filename and PCA mode.

    filename of BLOB will be '<INPUT>_<MODE>_dih.blob', where
    <INPUT> is the filename of the input without '.gro' suffix
    and mode is the PCA mode (e.g. 'cov' or 'corr').
    """
    if params["PCA_proj"]:
      projMode = "_proj" + "".join( map(str, params["PCA_proj"]) )
    else:
      projMode = ""

    if "PCA_cols" in params.keys():
      cols = "_colhash_" + hashlib.md5("".join( map(str, params["PCA_cols"]) )).hexdigest()[:8]

    return os.path.splitext(params["input"])[0] + "_" + params["PCA_mode"] + projMode + cols + "_pca.blob"

  def __init__(self, params=None, autosave=True ):
    if params:
      self.params = params
    else:
      self.params = Params()
    self.log = logging.getLogger( " PCA " )
    self.autosave = autosave
    self.params.setDefault( "PCA_mode", "cov" )
    self.params.setDefault( "PCA_proj", None  )
    # load PCA if available
    if "input" in self.params.keys():
      blobname = self.blobFilename( self.params )
      for filename in self.params["PCA_BLOBS"]:
        if ( os.path.basename(filename) == blobname ) and self.autosave:
          self.log.info( " loading existing PCA data from: " + filename )
          self.projection, self.eigenvals, self.eigenvecs, self.params["PCA_mode"] = blobLoad( filename )
          self.log.info( " ... finished." )
          break
      else:
        # compute PCA (if input file is given)
        if "input" in self.params.keys():
          m = GroFile( self.params ).readCoords()
          self.run( m )
          del m

  def run(self, m):
    """
    run the analysis
    """
    if not (self.params["PCA_mode"] == 'cov' or self.params["PCA_mode"] == 'corr' ):
      raise "unknown mode: " + str(mode)
    self.log.info( " computing PCA" )
    if "PCA_cols" in self.params.keys():
      # reduce matrix to given columns
      colFilter = [0]*max( self.params["PCA_cols"] )
      for i in self.params["PCA_cols"]:
        colFilter[i-1] = 1  # i-1: PCA_cols definition is given as base-1
      m = m.compress( colFilter, axis=1 )

    # calculate covariance matrix for given trajectory
    c = numpy.cov( m, rowvar=0 )
    if self.params["PCA_mode"] == 'corr':
      # standard deviations for different variables (i.e. columns)
      std = numpy.array( m.std(0) ).flatten()
      # renormalize covariance matrix
      # to get correlation matrix
      rows, cols = c.shape
      for i in range( rows ):
        for j in range( cols ):
          c.itemset( (i,j) , c.item( (i,j) )/(std[i]*std[j]) )
    # solve eigensystem:
    #  s = squared eigenvalues
    #  v = (conjugate transpose) matrix of right eigenvectors
    _ ,s,v = numpy.linalg.svd( c )
    del c
    v = v.transpose()

    if self.params["PCA_proj"]:
      nRows, nCols = v.shape
      # set indices to base0
      colsToKeep = map(lambda x: x-1, self.params["PCA_proj"])
      p = m*v
      # set columns of p not given in list to zero -> v'
      for i in range(nCols):
        if not (i in colsToKeep):
          p.T[i] = scipy.zeros( p.shape[0] )
      # project data on selected eigenvecs only
      self.projection = p*v.T
    else:
      self.projection = m*v
    self.eigenvals = s
    self.eigenvecs = v
    self.log.info( " ... finished." )
    if self.autosave:
      blobName = self.blobFilename(self.params)
      self.log.info( " saving PCA data to: " + blobName )
      blobDump( (self.projection, self.eigenvals, self.eigenvecs, self.params["PCA_mode"]),
                blobName
      )
      self.log.info( " ... finished." )
    return self

  def selectStructures(self, pos1, pos2, pc1, pc2=None):
    """
    select structures from projection (either 1D or 2D)

    - pos1: either float (1D) giving xMin or position (2D) with values pos1.xdata & pos1.ydata
    - pos2: either float (1D) giving xMax or position (2D) with values pos2.xdata & pos2.ydata
    - pc1:  id of PC plotted on x-axis
    - pc2:  id of PC plotted on y-axis (in case of 2D histograms)
    """
    if (type(pos1) == float) and (type(pos2) == float):
      # pos1 & pos2 are just x-values
      xMin = pos1
      xMax = pos2
      yMin = yMax = None
    else:
      # pos1 & pos2 are positions in 2D plane
      xMin = pos1.xdata
      xMax = pos2.xdata
      yMin = pos1.ydata
      yMax = pos2.ydata
    # check order
    xMin, xMax = minMax( xMin, xMax )
    yMin, yMax = minMax( yMin, yMax )
    # print short info
    if yMin and yMax:
      self.log.info( "calculating $\phi$ / $\psi$ distribution for selected data: x_min=%f, x_max=%f, y_min=%f, y_max=%f" %
                     (xMin, xMax, yMin, yMax)
      )
    else:
      self.log.info( "calculating $\phi$ / $\psi$ distribution for selected data: x_min=%f, x_max=%f" % (xMin, xMax))
    # get indices of selected structures
    selectedIds = []
    for i in range( self.projection.shape[0] ):
      # select from 1D space
      valuePC1 = self.projection.item( i, pc1 )
      inRange = (xMin <= valuePC1) and (valuePC1 <= xMax)
      if pc2:
        # additionally select from 2nd dimension
        valuePC2 = self.projection.item( i, pc2 )
        inRange = inRange and (yMin <= valuePC2) and (valuePC2 <= yMax)
      if inRange:
        selectedIds.append( i )
    return selectedIds


  def plot1DHist(self, pc, bins=200, prep=None):
    """
    plot 1d-histogram of projections of a PC

    **arguments**

      - pc:     id of the PC plotted
      - bins:   # of bins
      - prep:     if set, diagram will be saved to a file
                  called [prep]_[pc]_.eps.
                  if not set, diagram will show up interactively.
    """
    transp = self.projection.transpose()
    pcDat = numpy.array( transp[pc] ).flatten()

    plt.figure()
    ax = plt.subplot(211)
    plt.hist( x=pcDat, bins=bins, histtype='step' )
    plt.xlabel( str(pc+1) + ". PC" )
    plt.ylabel( "counts" )
    plt.title( "%s. PC" % str(pc+1) )

    self.redonePlots1D = 0
    def onselect( xMin, xMax ):
      #TODO: finish plotting of dihedral histograms for 1D cPCA hist
      pass

    span = matplotlib.widgets.SpanSelector( ax, onselect )


  def plot2DHist(self, pcX, pcY, bins=200, prep=None, mode='population', angles=None):
    """
    plot 2d-histogram of projections of one PC over the other

    **arguments**

      - pcX:      id of the PC plotted over the X-axis
      - pcY:      id of the PC plotted over the Y-axis
      - bins:     # of bins
      - prep:     if set, diagram will be saved to a file
                  called [prep]_[pcX]_[pcY].eps.
                  if not set, diagram will show up interactively.
      - mode:     either 'population' or 'fel' (for "free energy landscape")

    **parameters**

      - plot_dihedrals:  (DEFAULT: []) if set, only plot given dihedral pair
                         in histogram. (e.g.: [1,3,4] for PhiPsi1, PhiPsi3, PhiPsi4).
    """
    transp = self.projection.transpose()
    pc1 = numpy.array( transp[pcX] ).flatten()
    pc2 = numpy.array( transp[pcY] ).flatten()

    if angles:
      pc1, pc2 = self.filterAngles(pc1, pc2, angles)


    def replot( first=True ):
      fig = plt.figure()
      if first:
        ax = plt.subplot(111)
      else:
        ax = plt.subplot(211)

      hist, xedges, yedges = numpy.histogram2d(pc1, pc2, bins=bins)
      if mode == 'fel':
        # compute free energy landscape by taking ln of population
        hist = scipy.array(  map(lambda x: scipy.log(x), hist)  )
      extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
      plt.imshow(hist.T, extent=extent, interpolation='nearest', origin='lower', aspect='auto')
      plt.colorbar()
      plt.xlabel( str(pcX+1) + ". PC" )
      plt.ylabel( str(pcY+1) + ". PC" )
      if mode == 'population':
        #plt.title( "combined population of %s. and %s. PC" % (str(pcX+1), str(pcY+1)) )
        plt.title( "P($v_{%s}, v_{%s}$)" % (str(pcX+1), str(pcY+1)) )
      elif mode == 'fel':
        plt.title( "free energy landscape along %s. and %s. PC" % (str(pcX+1), str(pcY+1)) )
      else:
        raise "UnknownMode"
      self.redonePlots2D = 0
      self.axSubplot = None
      return ax

    ax = replot()

    def onselect(pos1,pos2):
      self.params.setDefault( "plot_dihedrals", [] )

      rectP1x = pos1.xdata
      rectP1y = pos1.ydata
      rectP2x = pos2.xdata
      rectP2y = pos2.ydata

      ax = replot( first=False )
      ax.add_patch( plt.Rectangle(
                      (rectP1x,rectP1y),
                       rectP2x-rectP1x,
                       rectP2y-rectP1y,
                       fill=False,
                       color='red',
                       linewidth=2.0
                    )
      )

      # get structure ids from selection
      filteredIds = self.selectStructures( pos1, pos2, pcX, pcY )
      self.log.info( "found %i structures in selected area" % len(filteredIds) )
      self.log.info( "first five structure ids: %s" % str(filteredIds[:5]) )

      if self.axSubplot:
        self.axSubplot.clear()
      self.axSubplot = plt.subplot(212)
      fontP = FontProperties()
      fontP.set_size('x-small')
      # get dihedral data for selected ids
      dih = Dihedrals( self.params )
      dihData = dih.filteredDihData( filteredIds )
      xRange = binning( -180, 180, bins )

      #colormap = 'spectral'
      colormap = 'jet'

      for i, dihIndex in enumerate(self.params["plot_dihedrals"]):
        self.redonePlots2D += 1
        h, _ = numpy.histogram( dihData["phi"][dihIndex], bins=bins, range=(-180,180) )
        #plt.plot( xRange, h.T, label="phi%i__%i" % (dihIndex, self.redonePlots2D) )
        plt.plot( xRange,
                  h.T,
                  label="phi%i" % (dihIndex+1), # +1, since phi / psi 2 is actually first group to plot
                  color=plt.get_cmap(colormap)( float(2*i)/(2*(len(self.params["plot_dihedrals"]))) )
        )

        h, _ = numpy.histogram( dihData["psi"][dihIndex], bins=bins, range=(-180,180) )
        #plt.plot( xRange, h.T, label="psi%i__%i" % (dihIndex, self.redonePlots2D) )
        plt.plot( xRange,
                  h.T,
                  label="psi%i" % (dihIndex+1),
                  color=plt.get_cmap(colormap)( float(2*i+1)/(2*(len(self.params["plot_dihedrals"]))) )
        )
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., prop=fontP)
      # plot the selected area
      plt.show()

    span = matplotlib.widgets.RectangleSelector( ax, onselect )
    plt.show()


  def filterAngles(self, pc1, pc2, angles):
    pc1 = pc1.tolist()
    pc2 = pc2.tolist()

    # load dihedrals
    dih = Dihedrals(self.params)
    angleFromToFilters = []
    for res in angles.keys():
      if (res[:3] == "phi") or (res[:3] == "psi"):
        n = int(res[3:])
        resName = res[:3]
        if resName == "phi":
          # -2 = -1 -1,  since phi2/psi2 are first groups (-1) AND lists in python start with index 0 (-1)
          n = 2*(n-2)
        elif resName == "psi":
          n = 2*(n-2) + 1
        else:
          raise "UnknownDihedralDefinition"

        angleFromToFilters.append( (n, angles[res][0], angles[res][1]) )

    def dihedralsInRange( rangeTuple, d ):
      n, aBottom, aTop = rangeTuple
      if (aBottom <= d[0,n]) and (d[0,n] <= aTop):
        return True
      else:
        return False

    for i in range( len(dih.dihedrals)-1, 0, -1 ):
      # if dihedral(s) not in given range(s): dismiss pc-data with same index
      d = dih.dihedrals[i]
      for t in angleFromToFilters:
        if not dihedralsInRange( t, d ):
          del pc1[i]
          del pc2[i]
          break

    return (scipy.array(pc1), scipy.array(pc2))




  def calculateConvergence(self):
    normalizedEigenvals = map( lambda x: x/sum(self.eigenvals), self.eigenvals )
    acc = [ 0.0 ]
    for i in range( len(normalizedEigenvals) ):
      acc.append( sum(normalizedEigenvals[:i+1]) )
    self.convergence = acc
    return self.convergence

  def plotConvergence(self, prep=None):
    conv = self.calculateConvergence()
    plt.figure()
    plt.subplot(111)
    plt.plot( range(len(conv)), conv, marker='o' )
    plt.xticks( range(len(conv)) )
    plt.xlabel( "index of PC" )
    plt.ylabel( "acc. percentage" )
    plt.title( "accumulated percentage of variation of PCs" )
    plt.show()

