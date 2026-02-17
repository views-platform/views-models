1 Find HPs comparable to baseline


Your starting architecture should be: 

* num_stacks=1, num_blocks=1, num_layers=2, layer_widths=64.

sue generic=true until you have the right hps 


input chunk: should be a lot longer
theory: generic = false and n stacks = 2 (or 3)


  * Aggressively increase `false_negative_weight`: {'distribution': 'uniform', 'min':
         4.0, 'max': 20.0}.




Reduce feature set to bare UCDP
Add ACLED

HP sweep again

Try shrinkage instead of huber .. 


