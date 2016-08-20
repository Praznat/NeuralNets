package modeler;

import ann.indirectencodings.RelationManager;
import modulemanagement.ModuleManagerPure;

public class ModelLearnerModularPure extends ModelLearnerHeavy {

	private final ModuleManagerPure moduleManager;
	private final RelationManager<Integer> relMngr;
	private int processTransitions;
	private int processTimes;

	public ModelLearnerModularPure(RelationManager<Integer> relMngr, ModuleManagerPure moduleManager,
			int maxReplaySize, int processTimes) {
		super(500, new int[] {}, null, null, null, maxReplaySize);
		this.relMngr = relMngr;
		this.processTransitions = maxReplaySize;
		this.moduleManager = moduleManager;
		this.processTimes = processTimes;
	}
	
	@Override
	public double[] newStateVars(double[] stateVars, double[] action, int jointAdjustments) {
		return moduleManager.getOutputs(this, relMngr, stateVars, action);
	}
	
	public void learn() {
		moduleManager.processFullModel(this, relMngr, processTransitions, processTimes);
	}
	
	public void learnGradually(int modules0, int modulesT, double scoreThresh0, double scoreThreshT, int steps) {
		processTimes = 1; // errrr..
		for (int i = 0; i <= steps; i++) {
			int m = modules0 + (int)Math.round(i * ((double)modulesT - modules0) / steps);
			double s = scoreThresh0 + (i * (scoreThreshT - scoreThresh0) / steps);
			moduleManager.setMaxModules(m);
			moduleManager.setMinScore(s);
			learn();
		}
	}
	
	@Override
	public void learnOnline(double lRate, double mRate, double sRate) {
		throw new IllegalStateException("User learn()");
	}
	
	@Override
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, long displayProgressMs, double stopAtErrThreshold) {
		throw new IllegalStateException("User learn()");
	}

	public ModuleManagerPure getModuleManager() {
		return moduleManager;
	}

	public RelationManager<Integer> getRelMngr() {
		return relMngr;
	}

	@Override
	public void feedForward() {
		throw new IllegalStateException("UNIMPLEMENTED METHOD");
	}

	@Override
	public double[] upJointOutput(double[] vars, int postStateIndex, int rounds) {
		throw new IllegalStateException("UNIMPLEMENTED METHOD");
	}

	@Override
	public ModelNeuralNet getTransitionsModule() {
		throw new IllegalStateException("NO GLOBAL NEURAL NETS FOR PURE MODULE LEARNER");
	}

	@Override
	public ModelNeuralNet getFamiliarityModule() {
		throw new IllegalStateException("NO GLOBAL NEURAL NETS FOR PURE MODULE LEARNER");
	}
}
