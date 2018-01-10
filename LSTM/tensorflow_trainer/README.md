Tensorflow Trainer

A selection of classes to make training with tensorflow a little easier. 

Usage:

This code uses argparse as a way to pass arguments. Each class defines its own arguments using the add_args method. This alows creation of one arg variable which can be passed to all classes. If you are unframiliar with argparse you might want to look up its syntax.

Extend two abstract classes
      1) abstract_classes/data_loader.py
      2) abstract_classes/network.py

create an optimizer with the desired optimization networks. EG.

       #create optimizer
       optimizer = Optimizer([MomentumWrapper,AdadeltaWrapper,AdamWrapper])

create a parser object from argparse and call add_arg methods

       #create parser
       parser = argparse.ArgumentParser(conflict_handler='resolve')

       #add arguments to pareser
       optimizer.add_args(parser)
       Trainer.add_args(parser)
       Network.add_args(parser)
       DataLoader.add_args(parser)

       #add any aditional args you might want yourself then call
       args = parser.parse_args()

Instance your network and dataloader

	 #create network and data loader
	 print(bcolors.OKGREEN + 'Creating DataLoader' + bcolors.ENDC)
    	 data_loader = DataLoader(args)
    	 print(bcolors.OKGREEN + 'Creating Network' + bcolors.ENDC)
    	 network = Network(args,VALUE_SIZE,POS_SIZE,NUM_CHILDREN_SIZE)

Create and call the trainer

       #create and run trainer
       print(bcolors.OKGREEN + 'Initalizing Trainer' + bcolors.ENDC)
       trainer = Trainer(args, args.run_name,data_loader,network,optimizer,shared_output_metrics=[accuracy],eval_output_metrics=[])
       trainer.train()
       print(bcolors.OKGREEN + 'Finished' + bcolors.ENDC)
