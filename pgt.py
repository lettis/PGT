#!/usr/bin/env python
import os
import sys
import tempfile
import logging
import cPickle
from glob import glob

try:
  import numpy
except:
  print "error: numpy not found. cannot do without it. sorry."
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


# set log-level to INFO
logging.basicConfig( level=logging.INFO )



#################
# definitions

# atomic masses [u]; from GROMACS 'atommass.dat'
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


#TODO: write INFO for blob dumping/loading
def blobDump( obj, filename ):
  """write serialized version of object to file"""
  fh = open( filename, "wb" )
  cPickle.dump( obj, fh, cPickle.HIGHEST_PROTOCOL )
  fh.close()
  log = logging.getLogger( " blobDump " )
  log.info( "dumping '%s'-object to %s" % (obj.__class__.__name__, filename) )

def blobLoad( filename ):
  """load object from file and return it"""
  fh = open( filename, "rb" )
  obj = cPickle.load( fh )
  fh.close()
  log = logging.getLogger( " blobLoad " )
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
  # split ref-structure into list of 3D-vectors
  vs_ref = []
  for i in range(nCols / 3):
    if not ref:
      x,y,z = M.item(0,3*i), M.item(0,3*i+1), M.item(0,3*i+2)
    else:
      x,y,z = ref.item(0,3*i), ref.item(0,3*i+1), ref.item(0,3*i+2)
    vs_ref.append( numpy.array([x,y,z]) )
  # calculate centroid for ref-structure
  centroidRef = numpy.zeros( 3 )
  for vec in vs_ref:
    centroidRef += vec
  # translate to center (reference structure)
  for i in range(len(vs_ref)):
    vs_ref[i] -= centroidRef
  # set up ref matrix for SVD fit
  Y = numpy.matrix(vs_ref).transpose()
  ## calculate rotation matrices
  M_fit = []
  for r in range(nRows):
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
    if ("DIH_BLOB" in self.params.keys()) and self.autosave:
      self.dihedrals = self.load() 
    elif "dih_input" in self.params.keys():
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
    blobDump( self.dihedrals, self.blobFilename(self.params["dih_input"]) )

  def load(self):
    """load data from BLOB-file"""
    return blobLoad( self.params["DIH_BLOB"] )
  
  def read(self):
    """
    read dihedrals from .gro-file (given as 'dih_input' in params)
    """
    # create unique tempfile for g_rama-output
    _, filename = tempfile.mkstemp(suffix=".xvg", prefix="dih_", dir=".")
    # delete the tempfile, just keep its name
    os.unlink( filename )
    # run g_rama to generate dihedrals
    cmd = "g_rama -f %s -s %s -o %s" % (self.params["dih_input"], self.params["topology"], filename)
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
    # parse file and close
    linewise( fh, perLine, buf )
    # remove tempfile
    os.unlink( filename )
    self.dihedrals = numpy.matrix( buf["dih"] )
    # if autosave enabled, save data to file
    if self.autosave:
      self.save()

  def plot1DHist(self, dih, nGroup, bins=200, prep=None):
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
    if dih == 'phi':
      offset = 0
    elif dih == 'psi':
      offset = 1
    else:
      raise "unknown angle: " + dih
    transp = self.dihedrals.transpose()
    dihDat = numpy.array( transp[2*nGroup + offset] ).flatten()
    plt.hist( x=dihDat, bins=bins, histtype='step' )
    #TODO: implement write to file


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
    phi = numpy.array( transp[ (nGroup-1)*2    ] ).flatten()
    psi = numpy.array( transp[ (nGroup-1)*2 +1 ] ).flatten()

    plt.figure()
    ax = plt.subplot( 211 )

    h, xedges, yedges = numpy.histogram2d( phi, psi, bins=bins )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    plt.imshow(h.T,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
    plt.colorbar()
    plt.xlabel( "phi%i" % nGroup )
    plt.ylabel( "psi%i" % nGroup )
    self.redonePlotsRama = 0
    def onselect(pos1, pos2):
      ax = plt.subplot( 212 )
      fontP = FontProperties()
      fontP.set_size('x-small')
      self.params.setDefault( "plot_dihedrals", [nGroup] )
      # pos1 & pos2 are positions in 2D plane
      xMin = pos1.xdata
      xMax = pos2.xdata
      yMin = pos1.ydata
      yMax = pos2.ydata
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
        self.redonePlotsRama += 1
        h, _ = numpy.histogram( dihData["phi"][dihIndex], bins=bins, range=(-180,180) )
        plt.plot( xRange, h.T, label="phi%i__%i" % (dihIndex, self.redonePlotsRama) )
        h, _ = numpy.histogram( dihData["psi"][dihIndex], bins=bins, range=(-180,180) )
        plt.plot( xRange, h.T, label="psi%i__%i" % (dihIndex, self.redonePlotsRama) )
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., prop=fontP)

    span = matplotlib.widgets.RectangleSelector( ax, onselect )


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
    if cartCoords:
      cCoords = cartCoords
    elif 'input' in self.params:
      cCoords = GroFile( self.params ).readCoords()
    else:
      raise Exception( 
        "unable to convert cartesian coordinates to internals: no cartesians given and no input specified in 'params'"
      )
    nRows, nAtoms = cCoords.shape
    # we assume 3D space, therefore the number of atoms is
    # 1/3 of cartesian coords (x1,y1,z1,x2,y2,z2,...xN,yN,zN)
    nAtoms /= 3
    coordsN = lambda i: numpy.array( [cCoords[3*i], cCoords[3*i+1], cCoords[3*i+2]] )
    self.bonds = []
    self.angles = []
    self.dihedrals = []
    for n in range( nRows ):
      bondsCurStruct = []
      anglesCurStruct = []
      dihedralsCurStruct = []
      for i in range( nAtoms-1 ):
        a = coordsN(i)
        b = coordsN(i+1)
        # calculate bond lengths of consecutive atoms
        bondsCurStruct.append( numpy.sqrt(numpy.dot(a,b)) )
        if i < nAtoms-2:
          # calculate angles between atoms
          c = coordsN(i+2)
          anglesCurStruct.append( numpy.math.acos(numpy.dot((a-b),(c-b))) )
        if i < nAtoms-3:
          # calculate dihedrals angles
          d = coordsN(i+3)
          b1 = b-a
          b2 = c-b
          b3 = d-c
          dihedralsCurStruct.append(
            numpy.atan2(
              numpy.math.sqrt(numpy.dot(b2,b2)) * b1 * numpy.cross(b2,b3),
              numpy.dot(numpy.cross(b1,b2),numpy.cross(b2,b3))
            )
          )
      self.bonds.append( bondsCurStruct )
      self.angles.append( anglesCurStruct )
      self.dihedrals.append( dihedralsCurStruct )
      


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
    return com / self.totalMass

  def radiusOfGyration(self):
    """calculate radius of gyration of structure"""

    totalMass = self.totalMass()

    #TODO: finish radius of gyration calculation (reference: zim -> bmd -> statistics)



  def __str__(self):
    """prints structure in simple 'x1 y1 z1 x2 y2 z2 ...' representation"""
    buf = []
    for atom in self.atoms:
      buf.extend( [atom.r[0], atom.r[1], atom.r[2]] )
    return " ".join( buf )
    

class Trajectory:
  """
  represents an MD trajectory (arbitrary coordinates)

  **attributes**

    - dt:          timestep of trajectory [ns]
    - t:           length of trajectory [ns]
    - structures:  list of structures
  """
  def __init__(self, dt=None, t=None):
    self.dt = None
    self.t  = None
    self.structures = []

  def filter( ids ):
    """keep only structures whose id is in the given list of ids"""
    self.structures = map(  lambda i,s: s,
                            filter( lambda i,s: i in ids,  enumerate(self.structures) )
    )


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


class Trjconv:
  """wrapper around the trjconv binary"""
  def __init__(self, params, projectDirectory=".", pathToBinary="trjconv"):
    # TODO: move 'projectDirectory' and 'pathToBinary' to params
    self.params = params
    self.projectDir = projectDirectory
    self.binary = pathToBinary

  def fit(self):
    """
    run fit on trajectory

    **parameters**

      - input:     trajectory filename for input
      - output:    trajectory filename for output
      - topology:  topology file 
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
    cmd +=               " -s " +   self.params["topology"]
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
  def __init__(self, filename=None):
    # TODO: move 'filename' to params
    self.groFile = filename
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
    self.log.info( " reading coords from trajectory: " + self.groFile )
    m = []
    fh = open( self.groFile, "r" )
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

  def read(self):
    """read .gro-file and return Trajectory object"""
    # TODO: currently ignores last line of a data-set.
    #       what is the meaning of this line?
    traj = Trajectory()
    #TODO: change to make use of 'linewise()'
    fh = open( self.groFile, "r" )
    try:
      while 1:
        try:
          line = fh.next()
          if 'Generated' in line:
            nAtoms = int( fh.next().strip() )
            # append next structure to trajectory
            traj.structures.append( Structure() )
            for i in range(nAtoms):
              line = fh.next().strip().split()
              # append atom to structure
              traj.structures[-1].atoms.append(
                Atom( atom    = line[1],
                      r       = map( float, [line[3], line[4], line[5]] ),
                      v       = map( float, [line[6], line[7], line[8]] ),
                      residue = line[0]
                )
              )
        except StopIteration:
          break # exit the while-loop
    finally:
      fh.close()
    return traj

  def write(self, traj, mode="w"):
    """
    write trajectory to .gro-file

    **arguments**

      - traj:   a Trajectory-object
      - mode:   'w' for overwrite (DEFAULT), 'a' for append
    """
    #TODO: finish
    pass



######################
# statistics
class PCA:
  """
  perform a principal component analysis on the given matrix

  **mode**

    perform PCA using either covariance- or correlation matrix.
    - mode='cov':  use covariance matrix (DEFAULT)
    - mode='corr': use correlation matrix

  **attributes**

    - projection: trajectory projected onto PCs
    - fracs:      fractions of variance for PCs
  """
  @staticmethod
  def blobFilename( inputFilename, pcaMode ):
    """
    generate filename for a PCA-BLOB based on an input filename and PCA mode.

    filename of BLOB will be '<INPUT>_<MODE>_dih.blob', where
    <INPUT> is the filename of the input without '.gro' suffix
    and mode is the PCA mode (e.g. 'cov' or 'corr').
    """
    return os.path.splitext(inputFilename)[0] + "_" + pcaMode + "_pca.blob"

  def __init__(self, params=None, autosave=True ):
    if params:
      self.params = params
    else:
      self.params = Params()
    self.log = logging.getLogger( " PCA " )
    self.autosave = autosave
    self.params.setDefault( "PCA_mode", "cov" )
    # load PCA if available
    if "input" in self.params.keys():
      blobname = self.blobFilename( self.params["input"], self.params["PCA_mode"] )
      for filename in self.params["PCA_BLOBS"]:
        if ( os.path.basename(filename) == blobname ) and self.autosave:
          self.log.info( " loading existing PCA data from: " + filename )
          self.projection, self.eigenvals, self.eigenvecs, self.params["PCA_mode"] = blobLoad( filename )
          self.log.info( " ... finished." )
          break
      else:
        # compute PCA (if input file is given)
        if "PATH" in self.params.keys():
          m = GroFile( self.params["PATH"] + "/" + self.params["input"] ).readCoords()
          self.run( m )
          del m

  def run(self, m):
    """
    run the analysis
    """
    if not (self.params["PCA_mode"] == 'cov' or self.params["PCA_mode"] == 'corr' ):
      raise "unknown mode: " + str(mode)
    self.log.info( " computing PCA" )
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

    self.projection = m*v
    self.eigenvals = s
    self.eigenvecs = v
    self.log.info( " ... finished." )
    if self.autosave and "PATH" in self.params.keys():
      blobName = self.params["PATH"] + "/" + self.blobFilename(self.params["input"], self.params["PCA_mode"])
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


  def plot2DHist(self, pcX, pcY, bins=200, prep=None):
    """
    plot 2d-histogram of projections of one PC over the other

    **arguments**

      - pcX:      id of the PC plotted over the X-axis
      - pcY:      id of the PC plotted over the Y-axis
      - bins:     # of bins
      - prep:     if set, diagram will be saved to a file
                  called [prep]_[pcX]_[pcY].eps.
                  if not set, diagram will show up interactively.

    **parameters**

      - plot_dihedrals:  (DEFAULT: []) if set, only plot given dihedral pair
                         in histogram. (e.g.: [1,3,4] for PhiPsi1, PhiPsi3, PhiPsi4).
   """
    transp = self.projection.transpose()
    pc1 = numpy.array( transp[pcX] ).flatten()
    pc2 = numpy.array( transp[pcY] ).flatten()

    def replot( first=True ):
      fig = plt.figure()
      if first:
        ax = plt.subplot(111)
      else:
        ax = plt.subplot(211)

      hist, xedges, yedges = numpy.histogram2d(pc1, pc2, bins=bins)
      extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
      plt.imshow(hist.T, extent=extent, interpolation='nearest', origin='lower', aspect='auto')
      plt.colorbar()
      plt.xlabel( str(pcX+1) + ". PC" )
      plt.ylabel( str(pcY+1) + ". PC" )
      plt.title( "combined population of %s. and %s. PC" % (str(pcX+1), str(pcY+1)) )

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
                  label="phi%i" % (dihIndex),
                  color=plt.get_cmap(colormap)( float(2*i)/(2*(len(self.params["plot_dihedrals"]))) )
        )

        h, _ = numpy.histogram( dihData["psi"][dihIndex], bins=bins, range=(-180,180) )
        #plt.plot( xRange, h.T, label="psi%i__%i" % (dihIndex, self.redonePlots2D) )
        plt.plot( xRange,
                  h.T,
                  label="psi%i" % (dihIndex),
                  color=plt.get_cmap(colormap)( float(2*i+1)/(2*(len(self.params["plot_dihedrals"]))) )
        )
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., prop=fontP)

    span = matplotlib.widgets.RectangleSelector( ax, onselect )

    #TODO: implement write to file

  def plotConvergence(self, prep=None):
    normalizedEigenvals = map( lambda x: x/sum(self.eigenvals), self.eigenvals )
    acc = [ 0.0 ]
    for i in range( len(normalizedEigenvals) ):
      acc.append( sum(normalizedEigenvals[:i+1]) )

    plt.figure()
    plt.subplot(111)
    plt.plot( range(len(acc)), acc, marker='o' )
    plt.xticks( range(len(acc)) )
    plt.xlabel( "index of PC" )
    plt.ylabel( "eigenval of PC" )
    plt.title( "accumulated percentage of variation of PCs" )

