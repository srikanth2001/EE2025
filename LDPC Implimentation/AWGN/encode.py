# !/usr/bin/env python3

#Implementation of regular LDPC code for a BSC

from numpy import *
import matplotlib.pyplot as plt 
import random
from numpy.random import shuffle, randint
from numpy.linalg import inv, det

img = load('../binary_image.npy')
lx, ly = len(img), len(img[0])
img = array(img).flatten()
size = lx*ly
n = 20
p, q = 4, 5 #weights of columns (w_c) and rows(w_r) respectively

def pc_matrix():
    """
    This function constructs a LDPC parity check matrix
    H. The algorithm follows Gallager's approach where we create
    p submatrices and stack them together. Reference: Turbo
    Coding for Satellite and Wireless Communications, section
    9,3.
    Note: the matrices computed from this algorithm will never
    have full rank. (Reference Gallager's Dissertation.) They
    will have rank = (number of rows - p + 1). To convert it
    to full rank, use the function get_full_rank_H_matrix
    """

    ratioTest = (n*1.0) / q
    if ratioTest%1 != 0:
      print('\nError in pc_matrix: The ', end='')
      print('ratio of inputs n/q must be a whole number.\n')
      return

    # First submatrix first:
    m = (n*p) / q  # number of rows in H matrix
    submatrix1 = zeros((int(m / p),n))
    for row in arange(int(m / p)):
      range1 = row*q
      range2 = (row+1)*q
      submatrix1[row,range1:range2] = 1
      H = submatrix1

    # Create the other submatrices and vertically stack them on.
    submatrixNum = 2
    newColumnOrder = arange(n)
    while submatrixNum <= p:
      submatrix = zeros((int(m / p),n))
      shuffle(newColumnOrder)

      for columnNum in arange(n):
        submatrix[:,columnNum] = \
                                 submatrix1[:,newColumnOrder[columnNum]]

      H = vstack((H,submatrix))
      submatrixNum = submatrixNum + 1

    # Double check the row weight and column weights.
    size = H.shape
    rows = size[0]
    cols = size[1]

    # Check the row weights.
    for rowNum in arange(rows):
      nonzeros = array(H[rowNum,:].nonzero())
      if nonzeros.shape[1] != q:
        print('Row', rowNum, 'has incorrect weight!')
        return

    # Check the column weights
    for columnNum in arange(cols):
      nonzeros = array(H[:,columnNum].nonzero())
      if nonzeros.shape[1] != p:
        print('Row', columnNum, 'has incorrect weight!')
        return

    return H

def swap_columns(a,b,arrayIn):
  """
  Swaps two columns in a matrix.
  """
  arrayOut = arrayIn.copy()
  arrayOut[:,a] = arrayIn[:,b]
  arrayOut[:,b] = arrayIn[:,a]
  return arrayOut

def move_row_to_bottom(i,arrayIn):
  """"
  Moves a specified row (just one) to the bottom of the matrix,
  then rotates the rows at the bottom up.
  For example, if we had a matrix with 5 rows, and we wanted to
  push row 2 to the bottom, then the resulting row order would be:
  1,3,4,5,2
  """
  arrayOut = arrayIn.copy()
  numRows = arrayOut.shape[0]
  # Push the specified row to the bottom.
  arrayOut[numRows-1] = arrayIn[i,:]
  # Now rotate the bottom rows up.
  index = 2
  while (numRows-index) >= i:
    arrayOut[numRows-index,:] = arrayIn[numRows-index+1]
    index = index + 1
  return arrayOut

def getSystematicGmatrix(GenMatrix):
  """
  This function finds the systematic form of the generator
  matrix GenMatrix. This form is G = [I P] where I is an identity
  matrix and P is the parity submatrix. If the GenMatrix matrix
  provided is not full rank, then dependent rows will be deleted.
  This function does not convert parity check (H) matrices to the
  generator matrix format. Use the function generator
  for that purpose.
  """
  tempArray = GenMatrix.copy()
  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0
  while i < limit:
    # Flag indicating that the row contains a non-zero entry
    found = False
    for j in arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found = True
        # Increment rank by 1
        rank = rank + 1
        # make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        break
    if found == True:
      for k in arange(0,numRows):
        if k == i: continue
        # Checking for 1's
        if tempArray[k, i] == 1:
          # add row i to row k
          tempArray[k,:] = tempArray[k,:] + tempArray[i,:]
          # Addition is mod2
          tempArray = tempArray.copy() % 2
          # All the entries above & below (i, i) are now 0
      i = i + 1
    if found == False:
      # push the row of 0s to the bottom, and move the bottom
      # rows up (sort of a rotation thing)
      tempArray = move_row_to_bottom(i,tempArray)
      # decrease limit since we just found a row of 0s
      limit -= 1
      # the rows below i are the dependent rows, which we discard
  G = tempArray[0:i,:]
  return G

def generator(H, verbose=False):
  """
  If given a parity check matrix H, this function returns a
  generator matrix G in the systematic form: G = [I P]
    where:  I is an identity matrix, size k x k
            P is the parity submatrix, size k x (n-k)
  If the H matrix provided is not full rank, then dependent rows
  will be deleted first.
  """
  # First, put the H matrix into the form H = [I|m] where:
  #   I is (n-k) x (n-k) identity matrix
  #   m is (n-k) x k
  # This part is just copying the algorithm from getSystematicGmatrix
  tempArray = getSystematicGmatrix(H)

  # Next, swap I and m columns so the matrix takes the forms [m|I].
  n      = tempArray.shape[1]
  k      = n - tempArray.shape[0]
  I_temp = tempArray[:,0:(n-k)]
  m      = tempArray[:,(n-k):n]

  newH   = concatenate((m,I_temp),axis=1)

  # Now the submatrix m is the transpose of the parity submatrix,
  # i.e. H is in the form H = [P'|I]. So G is just [I|P]
  G = concatenate((identity(k),m.T),axis=1)
  return G

def get_full_rank_H_matrix(H, verbose=False):
  """
  This function accepts a parity check matrix H and, if it is not
  already full rank, will determine which rows are dependent and
  remove them. The updated matrix will be returned.
  """
  tempArray = H.copy()
  if linalg.matrix_rank(tempArray) == tempArray.shape[0]:
    if verbose:
      print('Returning H; it is already full rank.')
    return tempArray

  numRows = tempArray.shape[0]
  numColumns = tempArray.shape[1]
  limit = numRows
  rank = 0
  i = 0

  # Create an array to save the column permutations.
  columnOrder = arange(numColumns).reshape(1,numColumns)

  # Create an array to save the row permutations. We just need
  # this to know which dependent rows to delete.
  rowOrder = arange(numRows).reshape(numRows,1)

  while i < limit:
    if verbose:
      print('In get_full_rank_H_matrix; i:', i)
      # Flag indicating that the row contains a non-zero entry
    found  = False
    for j in arange(i, numColumns):
      if tempArray[i, j] == 1:
        # Encountered a non-zero entry at (i, j)
        found =  True
        # Increment rank by 1
        rank = rank + 1
        # Make the entry at (i,i) be 1
        tempArray = swap_columns(j,i,tempArray)
        # Keep track of the column swapping
        columnOrder = swap_columns(j,i,columnOrder)
        break
    if found == True:
      for k in arange(0,numRows):
        if k == i: continue
        # Checking for 1's
        if tempArray[k, i] == 1:
          # Add row i to row k
          tempArray[k,:] = tempArray[k,:] + tempArray[i,:]
          # Addition is mod2
          tempArray = tempArray.copy() % 2
          # All the entries above & below (i, i) are now 0
      i = i + 1
    if found == False:
      # Push the row of 0s to the bottom, and move the bottom
      # rows up (sort of a rotation thing).
      tempArray = move_row_to_bottom(i,tempArray)
      # Decrease limit since we just found a row of 0s
      limit -= 1
      # Keep track of row swapping
      rowOrder = move_row_to_bottom(i,rowOrder)

  # Don't need the dependent rows
  finalRowOrder = rowOrder[0:i]

  # Reorder H, per the permutations taken above .
  # First, put rows in order, omitting the dependent rows.
  newNumberOfRowsForH = finalRowOrder.shape[0]
  newH = zeros((newNumberOfRowsForH, numColumns))
  for index in arange(newNumberOfRowsForH):
    newH[index,:] = H[finalRowOrder[index],:]

  # Next, put the columns in order.
  tempHarray = newH.copy()
  for index in arange(numColumns):
    newH[:,index] = tempHarray[:,columnOrder[0,index]]

  if verbose:
    print('original H.shape:', H.shape)
    print('newH.shape:', newH.shape)

  return newH

def encode(msg, G): #function for encoding
  k = G.shape[0]
  code = zeros((int(len(msg)*n/k)))
  i, j = 0, 0
  while i+k < len(msg):
      code[j:j+n] = (matmul(G.T, msg[i:i+k]))%2
      i += k
      j += n

  return code

H = pc_matrix()
G = generator(H)
code_img = encode(img, G)
savetxt("../encoded_bits.dat", code_img)
savetxt("../G.dat", G)
savetxt("../H.dat", H)