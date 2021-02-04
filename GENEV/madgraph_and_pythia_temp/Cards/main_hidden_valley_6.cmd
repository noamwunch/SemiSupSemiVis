! Pythia settings

! 1) Settings used in the main program
Random:setSeed               = on                ! allow you to pick seed
Random:seed                  = 0                 ! seed number (0 is a random seed that depends on the time Pythia is called)

! 2) Settings related to output in init(), next() and stat()
Init:showChangedSettings     = on                ! list changed settings
Init:showChangedParticleData = off               ! list changed particle data
Next:numberCount             = 500               ! print message every n events
Next:numberShowInfo          = 2                 ! print event information n times
Next:numberShowProcess       = 2                 ! print process record n times
Next:numberShowEvent         = 2                 ! print event record n times

! 3) Settings for the event generation process in the Pythia8 library
PartonLevel:MPI              = off               ! multiparton interactions
PartonLevel:ISR              = on                ! initial-state radiation
PartonLevel:FSR              = on                ! final-state radiation
HadronLevel:Hadronize        = on                ! hadronization

! 4) Hidden valley settings

! Fragmentation and showering information
HiddenValley:fragment        = on                ! fragmentation for hidden valley particles
HiddenValley:FSR             = on                ! final state radiation for hidden valley particles (initial state radiation obviously doesn't exist)

! Spin information
HiddenValley:spinFv          = 2                 ! spin of mediator
HiddenValley:spinqv          = 1                 ! spin of dark quark (Note: if FV is a boson than qv is automatically assigned a spin of 1/2)

! Running information
HiddenValley:alphaOrder      = 1                 ! whether to (1) run alphaHV or (0) not
HiddenValley:nFlav           = Removenfl         ! number of dark flavors
HiddenValley:Ngauge          = RemovenGauge      ! number of dark QCD colours
HiddenValley:Lambda          = RemovelambdaD     ! dark confinement scale (should be larger than the dark quark mass, e.g. 1.1 * 4900101:m0)
HiddenValley:pTminFSR        = RemovepTmin	     ! pT cutoff for dark shower

! Vector meson information
HiddenValley:probVector      = RemoveProbV       ! probability to create vector mesons (the default value of 0.75 is what is expected from a naive spin count)

! Mass information
4900101:m0                   = RemoveDquarkMass  ! dark quark mass
4900111:m0                   = RemoveDpionMass   ! dark pion mass
4900113:m0                   = RemoveDRhoMass    ! dark rho mass

! Decay length information
4900111:tau0                 = RemovedecayLength ! dark scalar (pion) lifetime (in mm)

! Branching ratio information
4900111:0:all on rinv 102 12 -12                 ! dark pion decay to neutrinos
4900111:addchannel on rvis 102 1 -1     	     ! dark pion decay to down quarks
4900113:0:all on 1.000 102 4900111 4900111       ! dark vector to dark pions 100%
