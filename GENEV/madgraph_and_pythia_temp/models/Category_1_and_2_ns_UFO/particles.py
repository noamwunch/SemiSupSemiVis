# This file was automatically created by FeynRules 2.3.13
# Mathematica version: 9.0 for Microsoft Windows (64-bit) (January 25, 2013)
# Date: Mon 7 Aug 2017 20:30:22


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

import propagators as Prop

a = Particle(pdg_code = 22,
             name = 'a',
             antiname = 'a',
             spin = 3,
             color = 1,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'a',
             antitexname = 'a',
             charge = 0,
             LeptonNumber = 0,
             Y = 0)

Z = Particle(pdg_code = 23,
             name = 'Z',
             antiname = 'Z',
             spin = 3,
             color = 1,
             mass = Param.MZ,
             width = Param.WZ,
             texname = 'Z',
             antitexname = 'Z',
             charge = 0,
             LeptonNumber = 0,
             Y = 0)

W__plus__ = Particle(pdg_code = 24,
                     name = 'W+',
                     antiname = 'W-',
                     spin = 3,
                     color = 1,
                     mass = Param.MW,
                     width = Param.WW,
                     texname = 'W+',
                     antitexname = 'W-',
                     charge = 1,
                     LeptonNumber = 0,
                     Y = 0)

W__minus__ = W__plus__.anti()

g = Particle(pdg_code = 21,
             name = 'g',
             antiname = 'g',
             spin = 3,
             color = 8,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'g',
             antitexname = 'g',
             charge = 0,
             LeptonNumber = 0,
             Y = 0)

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 've',
              antitexname = 've~',
              charge = 0,
              LeptonNumber = 1,
              Y = 0)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code = 14,
              name = 'vm',
              antiname = 'vm~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vm',
              antitexname = 'vm~',
              charge = 0,
              LeptonNumber = 1,
              Y = 0)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code = 16,
              name = 'vt',
              antiname = 'vt~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vt',
              antitexname = 'vt~',
              charge = 0,
              LeptonNumber = 1,
              Y = 0)

vt__tilde__ = vt.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = Param.Me,
                      width = Param.ZERO,
                      texname = 'e-',
                      antitexname = 'e+',
                      charge = -1,
                      LeptonNumber = 1,
                      Y = 0)

e__plus__ = e__minus__.anti()

mu__minus__ = Particle(pdg_code = 13,
                       name = 'mu-',
                       antiname = 'mu+',
                       spin = 2,
                       color = 1,
                       mass = Param.MMU,
                       width = Param.ZERO,
                       texname = 'mu-',
                       antitexname = 'mu+',
                       charge = -1,
                       LeptonNumber = 1,
                       Y = 0)

mu__plus__ = mu__minus__.anti()

ta__minus__ = Particle(pdg_code = 15,
                       name = 'ta-',
                       antiname = 'ta+',
                       spin = 2,
                       color = 1,
                       mass = Param.MTA,
                       width = Param.ZERO,
                       texname = 'ta-',
                       antitexname = 'ta+',
                       charge = -1,
                       LeptonNumber = 1,
                       Y = 0)

ta__plus__ = ta__minus__.anti()

u = Particle(pdg_code = 2,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = Param.MU,
             width = Param.ZERO,
             texname = 'u',
             antitexname = 'u~',
             charge = 2/3,
             LeptonNumber = 0,
             Y = 0)

u__tilde__ = u.anti()

c = Particle(pdg_code = 4,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = Param.MC,
             width = Param.ZERO,
             texname = 'c',
             antitexname = 'c~',
             charge = 2/3,
             LeptonNumber = 0,
             Y = 0)

c__tilde__ = c.anti()

t = Particle(pdg_code = 6,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = Param.MT,
             width = Param.WT,
             texname = 't',
             antitexname = 't~',
             charge = 2/3,
             LeptonNumber = 0,
             Y = 0)

t__tilde__ = t.anti()

d = Particle(pdg_code = 1,
             name = 'd',
             antiname = 'd~',
             spin = 2,
             color = 3,
             mass = Param.MD,
             width = Param.ZERO,
             texname = 'd',
             antitexname = 'd~',
             charge = -1/3,
             LeptonNumber = 0,
             Y = 0)

d__tilde__ = d.anti()

s = Particle(pdg_code = 3,
             name = 's',
             antiname = 's~',
             spin = 2,
             color = 3,
             mass = Param.MS,
             width = Param.ZERO,
             texname = 's',
             antitexname = 's~',
             charge = -1/3,
             LeptonNumber = 0,
             Y = 0)

s__tilde__ = s.anti()

b = Particle(pdg_code = 5,
             name = 'b',
             antiname = 'b~',
             spin = 2,
             color = 3,
             mass = Param.MB,
             width = Param.ZERO,
             texname = 'b',
             antitexname = 'b~',
             charge = -1/3,
             LeptonNumber = 0,
             Y = 0)

b__tilde__ = b.anti()

H = Particle(pdg_code = 25,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = Param.MH,
             width = Param.WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             LeptonNumber = 0,
             Y = 0)

G0 = Particle(pdg_code = 250,
              name = 'G0',
              antiname = 'G0',
              spin = 1,
              color = 1,
              mass = Param.MZ,
              width = Param.WZ,
              texname = 'G0',
              antitexname = 'G0',
              goldstone = True,
              charge = 0,
              LeptonNumber = 0,
              Y = 0)

G__plus__ = Particle(pdg_code = 251,
                     name = 'G+',
                     antiname = 'G-',
                     spin = 1,
                     color = 1,
                     mass = Param.MW,
                     width = Param.WW,
                     texname = 'G+',
                     antitexname = 'G-',
                     goldstone = True,
                     charge = 1,
                     LeptonNumber = 0,
                     Y = 0)

G__minus__ = G__plus__.anti()

XFnu = Particle(pdg_code = 8000001,
                name = 'XFnu',
                antiname = 'XFnu~',
                spin = 2,
                color = 1,
                mass = Param.MXFnu,
                width = Param.WXFnu,
                texname = 'XFnu',
                antitexname = 'XFnu~',
                charge = 0,
                LeptonNumber = 1,
                Y = 0)

XFnu__tilde__ = XFnu.anti()

XFeL = Particle(pdg_code = 8000002,
                name = 'XFeL',
                antiname = 'XFeL~',
                spin = 2,
                color = 1,
                mass = Param.MXFeL,
                width = Param.WXFeL,
                texname = 'XFeL',
                antitexname = 'XFeL~',
                charge = -1,
                LeptonNumber = 1,
                Y = 0)

XFeL__tilde__ = XFeL.anti()

XFeR = Particle(pdg_code = 8000003,
                name = 'XFeR',
                antiname = 'XFeR~',
                spin = 2,
                color = 1,
                mass = Param.MXFeR,
                width = Param.WXFeR,
                texname = 'XFeR',
                antitexname = 'XFeR~',
                charge = -1,
                LeptonNumber = 1,
                Y = 0)

XFeR__tilde__ = XFeR.anti()

XFuL = Particle(pdg_code = 8000004,
                name = 'XFuL',
                antiname = 'XFuL~',
                spin = 2,
                color = 3,
                mass = Param.MXFuL,
                width = Param.WXFuL,
                texname = 'XFuL',
                antitexname = 'XFuL~',
                charge = 2/3,
                LeptonNumber = 0,
                Y = 0)

XFuL__tilde__ = XFuL.anti()

XFdL = Particle(pdg_code = 8000005,
                name = 'XFdL',
                antiname = 'XFdL~',
                spin = 2,
                color = 3,
                mass = Param.MXFdL,
                width = Param.WXFdL,
                texname = 'XFdL',
                antitexname = 'XFdL~',
                charge = -1/3,
                LeptonNumber = 0,
                Y = 0)

XFdL__tilde__ = XFdL.anti()

XFuR = Particle(pdg_code = 8000006,
                name = 'XFuR',
                antiname = 'XFuR~',
                spin = 2,
                color = 3,
                mass = Param.MXFuR,
                width = Param.WXFuR,
                texname = 'XFuR',
                antitexname = 'XFuR~',
                charge = 2/3,
                LeptonNumber = 0,
                Y = 0)

XFuR__tilde__ = XFuR.anti()

XFdR = Particle(pdg_code = 8000007,
                name = 'XFdR',
                antiname = 'XFdR~',
                spin = 2,
                color = 3,
                mass = Param.MXFdR,
                width = Param.WXFdR,
                texname = 'XFdR',
                antitexname = 'XFdR~',
                charge = -1/3,
                LeptonNumber = 0,
                Y = 0)

XFdR__tilde__ = XFdR.anti()

ns = Particle(pdg_code = 4900101,
              name = 'ns',
              antiname = 'ns~',
              spin = 1,
              color = 1,
              mass = Param.Mns,
              width = Param.Wns,
              texname = 'ns',
              antitexname = 'ns~',
              charge = 0,
              LeptonNumber = 0,
              Y = 0)

ns__tilde__ = ns.anti()

neut = Particle(pdg_code = 8001234,
                name = 'neut',
                antiname = 'neut',
                spin = 2,
                color = 1,
                mass = Param.Mneut,
                width = Param.Wneut,
                texname = 'neut',
                antitexname = 'neut',
                charge = 0,
                LeptonNumber = 0,
                Y = 0)

