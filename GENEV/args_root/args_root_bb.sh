#!/bin/bash

root_inp_file_dir="/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"
txt_out_file_dir="/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"

root_macro="root_tree_to_txt_with_rave_primvert"

dijet="true"
veto_isolep="false"
PT_min=50
PT_max=100000
Eta_min=-2.5
Eta_max=2.5
Mjj_min=1000
Mjj_max=100000
dR_jet=0.7
ystar_max=1.0
bkg_PID=5