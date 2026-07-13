from edelweissfe.utils.inputlanguage import Module

module = Module("discreterigidbody", "A discrete rigid body entity.")
module.addRequiredArg("nSet", "The node set containing the surface nodes of the rigid body.", str)
module.addRequiredArg("referencePoint", "The node set containing the single reference point.", str)
module.addOptionalArg("mass", "The mass of the rigid body.", float, None)
module.addOptionalArg("inertia", "The inertia tensor of the rigid body.", list, None)
module.addOptionalArg("initial_velocity", "The initial velocity vector.", list, None)
keyword = "discreterigidbody"
