import numpy as np

def loadData( fileName ):
    rawData = []
    with open(fileName) as f:
        for line in f:
            rawData.append(line.split(','))
    print '>> ', fileName, 'loaded...'
    npData = np.array(rawData)
    status = npData[ :, 1:4]
    answer = npData[ :, -1]
    mask = np.ones(npData.shape[1], dtype=bool)
    mask[1:4] = False  # string feature
    mask[-1]  = False  # label
    mask[19]  = False  # this column is all zero = =`
    other = npData[:,mask].astype('float')
    return status, other, answer


def loadClass( fileName ):
    d = { 'normal.\n': 0 }
    with open( fileName ) as f:
        for line in f:
            token = line.split()
            if token[1] == 'dos':
                d[token[0]+'.\n'] = 1
            elif token[1] == 'u2r':
                d[token[0]+'.\n'] = 2
            elif token[1] == 'r2l':
                d[token[0]+'.\n'] = 3
            else: 
                d[token[0]+'.\n'] = 4
    return d

def loadAttack( fileName ):
    d = { 'normal.\n': 0 }
    ii= 0
    with open( fileName ) as f:
        for line in f:
            ii += 1
            token = line.split()
            d[token[0]+'.\n'] = ii
    return d

def attack2class( Y ):
    for ii in range(Y.shape[0]):
        if   Y[ii] == 0: continue
        elif Y[ii] <=11: Y[ii] = 1
        elif Y[ii] <=19: Y[ii] = 2
        elif Y[ii] <=34: Y[ii] = 3
        else: Y[ii] = 4
    return

def loadTest( fileName ):
    rawData = []
    with open(fileName) as f:
        for line in f:
            rawData.append(line.split(','))
    print '>> ', fileName, 'loaded...'
    npData = np.array(rawData)
    status = npData[ :, 1:4]
    mask = np.ones(npData.shape[1], dtype=bool)
    mask[1:4] = False  # string feature
    mask[19]  = False  # this column is all zero = =`
    other = npData[:,mask].astype('float')
    return status, other

def outputCSV(fileName, Y):
    oFile = open(fileName, 'w')
    oFile.write('id,label\n')
    for i in range(Y.shape[0]):
        oFile.write('%d,%d\n' %(i+1,Y[i]))
    oFile.close()
    print '>> writing output file:', fileName
    return


