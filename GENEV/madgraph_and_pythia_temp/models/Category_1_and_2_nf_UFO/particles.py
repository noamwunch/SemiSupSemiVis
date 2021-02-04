# This file was automatically created by FeynRules 2.3.13
# Mathematica version: 9.0 for Microsoft Windows (64-bit) (January 25, 2013)
# Date: Mon 7 Aug 2017 21:15:30


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

XSnu = Particle(pdg_code = 7000001,
                name = 'XSnu',
                antiname = 'XSnu~',
                spin = 1,
                color = 1,
                mass = Param.MXSnu,
                width = Param.WXSnu,
                texname = 'XSnu',
                antitexname = 'XSnu~',
                charge = 0,
                LeptonNumber = 1,
                Y = 0)

XSnu__tilde__ = XSnu.anti()

XSeL = Particle(pdg_code = 7000002,
                name = 'XSeL',
                antiname = 'XSeL~',
                spin = 1,
                color = 1,
                mass = Param.MXSeL,
                width = Param.WXSeL,
                texname = 'XSeL',
                antitexname = 'XSeL~',
                charge = -1,
                LeptonNumber = 1,
                Y = 0)

XSeL__tilde__ = XSeL.anti()

XSeR = Particle(pdg_code = 7000003,
                name = 'XSeR',
                antiname = 'XSeR~',
                spin = 1,
                color = 1,
                mass = Param.MXSeR,
                width = Param.WXSeR,
                texname = 'XSeR',
                antitexname = 'XSeR~',
                charge = -1,
                LeptonNumber = 1,
                Y = 0)

XSeR__tilde__ = XSeR.anti()

XSuL = Particle(pdg_code = 7000004,
                name = 'XSuL',
                antiname = 'XSuL~',
                spin = 1,
                color = 3,
                mass = Param.MXSuL,
                width = Param.WXSuL,
                texname = 'XSuL',
                antitexname = 'XSuL~',
                charge = 2/3,
                LeptonNumber = 0,
                Y = 0)

XSuL__tilde__ = XSuL.anti()

XSdL = Particle(pdg_code = 7000005,
                name = 'XSdL',
                antiname = 'XSdL~',
                spin = 1,
                color = 3,
                mass = Param.MXSdL,
                width = Param.WXSdL,
                texname = 'XSdL',
                antitexname = 'XSdL~',
                charge = -1/3,
                LeptonNumber = 0,
                Y = 0)

XSdL__tilde__ = XSdL.anti()

XSuR = Particle(pdg_code = 7000006,
                name = 'XSuR',
                antiname = 'XSuR~',
                spin = 1,
                color = 3,
                mass = Param.MXSuR,
                width = Param.WXSuR,
                texname = 'XSuR',
                antitexname = 'XSuR~',
                charge = 2/3,
                LeptonNumber = 0,
                Y = 0)

XSuR__tilde__ = XSuR.anti()

XSdR = Particle(pdg_code = 7000007,
                name = 'XSdR',
                antiname = 'XSdR~',
                spin = 1,
                color = 3,
                mass = Param.MXSdR,
                width = Param.WXSdR,
                texname = 'XSdR',
                antitexname = 'XSdR~',
                charge = -1/3,
                LeptonNumber = 0,
                Y = 0)

XSdR__tilde__ = XSdR.anti()

nf = Particle(pdg_code = 4900101,
              name = 'nf',
              antiname = 'nf~',
              spin = 2,
              color = 1,
              mass = Param.Mnf,
              width = Param.Wnf,
              texname = 'nf',
              antitexname = 'nf~',
              charge = 0,
              LeptonNumber = 0,
              Y = 0)

nf__tilde__ = nf.anti()

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

