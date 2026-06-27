#!/usr/bin/env bash
cd /home/jaka6039/MT/assignment2/data/output_whole
/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/bin/extractor --sctype BLEU --scconfig case:true  --scfile run8.scores.dat --ffile run8.features.dat -r /home/jaka6039/MT/assignment2/data/europarl.dev.tk.lc.en -n run8.best100.out.gz
