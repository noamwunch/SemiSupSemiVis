# This file was automatically created by FeynRules 2.3.13
# Mathematica version: 9.0 for Microsoft Windows (64-bit) (January 25, 2013)
# Date: Mon 7 Aug 2017 20:30:22



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
cabi = Parameter(name = 'cabi',
                 nature = 'external',
                 type = 'real',
                 value = 0.227736,
                 texname = '\\theta _c',
                 lhablock = 'CKMBLOCK',
                 lhacode = [ 1 ])

lambdaXFl1 = Parameter(name = 'lambdaXFl1',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFl1}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 1 ])

lambdaXFl2 = Parameter(name = 'lambdaXFl2',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFl2}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 2 ])

lambdaXFl3 = Parameter(name = 'lambdaXFl3',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFl3}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 3 ])

lambdaXFeR1 = Parameter(name = 'lambdaXFeR1',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFeR1}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 4 ])

lambdaXFeR2 = Parameter(name = 'lambdaXFeR2',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFeR2}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 5 ])

lambdaXFeR3 = Parameter(name = 'lambdaXFeR3',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFeR3}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 6 ])

lambdaXFq1 = Parameter(name = 'lambdaXFq1',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFq1}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 7 ])

lambdaXFq2 = Parameter(name = 'lambdaXFq2',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFq2}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 8 ])

lambdaXFq3 = Parameter(name = 'lambdaXFq3',
                       nature = 'external',
                       type = 'real',
                       value = 1,
                       texname = '\\text{lambdaXFq3}',
                       lhablock = 'LAMBDADECAYXFERMION',
                       lhacode = [ 9 ])

lambdaXFuR1 = Parameter(name = 'lambdaXFuR1',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFuR1}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 10 ])

lambdaXFuR2 = Parameter(name = 'lambdaXFuR2',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFuR2}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 11 ])

lambdaXFuR3 = Parameter(name = 'lambdaXFuR3',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFuR3}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 12 ])

lambdaXFdR1 = Parameter(name = 'lambdaXFdR1',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFdR1}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 13 ])

lambdaXFdR2 = Parameter(name = 'lambdaXFdR2',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFdR2}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 14 ])

lambdaXFdR3 = Parameter(name = 'lambdaXFdR3',
                        nature = 'external',
                        type = 'real',
                        value = 1,
                        texname = '\\text{lambdaXFdR3}',
                        lhablock = 'LAMBDADECAYXFERMION',
                        lhacode = [ 15 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymdo = Parameter(name = 'ymdo',
                 nature = 'external',
                 type = 'real',
                 value = 0.00504,
                 texname = '\\text{ymdo}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 1 ])

ymup = Parameter(name = 'ymup',
                 nature = 'external',
                 type = 'real',
                 value = 0.00255,
                 texname = '\\text{ymup}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 2 ])

yms = Parameter(name = 'yms',
                nature = 'external',
                type = 'real',
                value = 0.101,
                texname = '\\text{yms}',
                lhablock = 'YUKAWA',
                lhacode = [ 3 ])

ymc = Parameter(name = 'ymc',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = '\\text{ymc}',
                lhablock = 'YUKAWA',
                lhacode = [ 4 ])

ymb = Parameter(name = 'ymb',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = '\\text{ymb}',
                lhablock = 'YUKAWA',
                lhacode = [ 5 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

yme = Parameter(name = 'yme',
                nature = 'external',
                type = 'real',
                value = 0.000511,
                texname = '\\text{yme}',
                lhablock = 'YUKAWA',
                lhacode = [ 11 ])

ymm = Parameter(name = 'ymm',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{ymm}',
                lhablock = 'YUKAWA',
                lhacode = [ 13 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MMU = Parameter(name = 'MMU',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{MMU}',
                lhablock = 'MASS',
                lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.00255,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.27,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00504,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.101,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

MXFnu = Parameter(name = 'MXFnu',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFnu}',
                  lhablock = 'MASS',
                  lhacode = [ 8000001 ])

MXFeL = Parameter(name = 'MXFeL',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFeL}',
                  lhablock = 'MASS',
                  lhacode = [ 8000002 ])

MXFeR = Parameter(name = 'MXFeR',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFeR}',
                  lhablock = 'MASS',
                  lhacode = [ 8000003 ])

MXFuL = Parameter(name = 'MXFuL',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFuL}',
                  lhablock = 'MASS',
                  lhacode = [ 8000004 ])

MXFdL = Parameter(name = 'MXFdL',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFdL}',
                  lhablock = 'MASS',
                  lhacode = [ 8000005 ])

MXFuR = Parameter(name = 'MXFuR',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFuR}',
                  lhablock = 'MASS',
                  lhacode = [ 8000006 ])

MXFdR = Parameter(name = 'MXFdR',
                  nature = 'external',
                  type = 'real',
                  value = 300,
                  texname = '\\text{MXFdR}',
                  lhablock = 'MASS',
                  lhacode = [ 8000007 ])

Mns = Parameter(name = 'Mns',
                nature = 'external',
                type = 'real',
                value = 5,
                texname = '\\text{Mns}',
                lhablock = 'MASS',
                lhacode = [ 4900101 ])

Mneut = Parameter(name = 'Mneut',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Mneut}',
                  lhablock = 'MASS',
                  lhacode = [ 8001234 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00407,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

WXFnu = Parameter(name = 'WXFnu',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFnu}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000001 ])

WXFeL = Parameter(name = 'WXFeL',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFeL}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000002 ])

WXFeR = Parameter(name = 'WXFeR',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFeR}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000003 ])

WXFuL = Parameter(name = 'WXFuL',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFuL}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000004 ])

WXFdL = Parameter(name = 'WXFdL',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFdL}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000005 ])

WXFuR = Parameter(name = 'WXFuR',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFuR}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000006 ])

WXFdR = Parameter(name = 'WXFdR',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{WXFdR}',
                  lhablock = 'DECAY',
                  lhacode = [ 8000007 ])

Wns = Parameter(name = 'Wns',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{Wns}',
                lhablock = 'DECAY',
                lhacode = [ 4900101 ])

Wneut = Parameter(name = 'Wneut',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Wneut}',
                  lhablock = 'DECAY',
                  lhacode = [ 8001234 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.sin(cabi)',
                   texname = '\\text{CKM1x2}')

CKM1x3 = Parameter(name = 'CKM1x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM1x3}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '-cmath.sin(cabi)',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM2x2}')

CKM2x3 = Parameter(name = 'CKM2x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM2x3}')

CKM3x1 = Parameter(name = 'CKM3x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x1}')

CKM3x2 = Parameter(name = 'CKM3x2',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '1',
                   texname = '\\text{CKM3x3}')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = 'M_W')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - sw2)',
               texname = 'c_w')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(sw2)',
               texname = 's_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw)/ee',
                texname = '\\text{vev}')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*vev**2)',
                texname = '\\text{lam}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vev',
               texname = '\\text{yb}')

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/vev',
               texname = '\\text{yc}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/vev',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(yme*cmath.sqrt(2))/vev',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymm*cmath.sqrt(2))/vev',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/vev',
               texname = '\\text{ys}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/vev',
                 texname = '\\text{ytau}')

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '(ymup*cmath.sqrt(2))/vev',
                texname = '\\text{yup}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu')

