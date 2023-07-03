I am a lazy individual,
so I do not like getting up from my chair and walking
across the room to flip my light switches when I am about to
watch a movie!

As a result, this is a 2-part speech recognition system:
* A raspberry PI 4B listens and recognizes speech commands
* A microcontroller listens to the raspberry PI and uses 
servos to flip the switch I ask for
  
Specifically, I would first say "Arduino" and then on successful
recognition of the word, there will be a time window 
where I specify which light I want to switch.

For example, I start by saying "Arduino" and then I can say
"Stairs", "Bar", or "Gym" and have those switches flipped!

The idea is to train a neural network on a more powerful
machine, and then use the tensorflow lite framework
to allow the raspberry pi to run inference.

For more specific information, look at the READMEs in the 
PiSystem and MicrocontrollerSystem directories

