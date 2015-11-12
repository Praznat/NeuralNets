package modulemanagement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;

import ann.Utils;
import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modeler.TransitionMemory;

public class ModuleManagerPure extends ModuleManager<Integer> {
	
	private static final double MIN_SCORE = 0.98;

	private static final int[] numHidden = new int[] {20};
	private final double lRate;
	private final double mRate;
	private final double sRate;
	private static final int epochs = 20;

	
	private int maxCrunchPerRound;
	private List<Integer> shuffledOutK;
	private int numTransitions;
	private ModelLearner loadedModeler;
	private Map<ModelLearner, TreeSet<OutputScore<Integer>>> oldOutputScores
		= new HashMap<ModelLearner, TreeSet<OutputScore<Integer>>>();
	private Map<ModelLearner, Map<Integer, ModuleDistribution<Integer>>> oldNodeModules
		= new HashMap<ModelLearner, Map<Integer, ModuleDistribution<Integer>>>();
	
	/** maxCrunchPerRound specifies how many new modules you can learn in a round */
	// TODO have max number of modules?
	public ModuleManagerPure(int maxCrunchPerRound, double lRate, double mRate, double sRate) {
		this.maxCrunchPerRound = maxCrunchPerRound;
		this.lRate = lRate;
		this.mRate = mRate;
		this.sRate = sRate;
	}
	
	/** return output activations given state and action inputs 
	 * @param numTransitions */
	/* modeler is only used to store experience */
	public double[] getOutputs(ModelLearner modeler, RelationManager<Integer> relMngr,
			double[] stateVars, double[] action) {
		loadModeler(modeler);
		double[] fullInputs = Utils.concat(stateVars, action);
		double[] fullOutputs = new double[stateVars.length];
		shuffleOutK(fullOutputs);
		AtomicInteger crunchesLeft = new AtomicInteger(maxCrunchPerRound);
		outputScores.clear(); // store them later in this method

		// look through output vars in random order
		// get prediction for each one
		for (int outK : shuffledOutK) {
			ModuleDistribution<Integer> moduleDistribution = nodeModules.get(outK);
			if (moduleDistribution == null) nodeModules.put(outK, moduleDistribution = new ModuleDistribution<Integer>());
			fullOutputs[outK] = getOutput(outK, moduleDistribution, fullInputs, crunchesLeft, modeler, relMngr);
		}
		return fullOutputs;
	}

	/** returns a predicted output */
	private double getOutput(int outK, ModuleDistribution<Integer> moduleDistribution, double[] fullInputs,
			AtomicInteger crunchesLeft, ModelLearner modeler, RelationManager<Integer> relMngr) {
		ReusableIntModule mostLikelyModule = (ReusableIntModule) moduleDistribution.getMostLikelyModule();
		// if no modules in distribution for this output, have to learn association ... (what about score?)
		if (mostLikelyModule == null) {
			mostLikelyModule = learnModuleAssociation(outK, modeler, relMngr, crunchesLeft, moduleDistribution);
		}
		// once you have a module, make the prediction
		double predOut = mostLikelyModule.evaluateOutput(outK, relMngr, fullInputs);
		double score = moduleDistribution.getHighestScore();
		// remember the score for this output
		addOutputScore(outK, score, outK);
//		System.out.println(outK + "	" + score + "	" + moduleDistribution.size());
		return predOut;
	}

	private ReusableIntModule learnModuleAssociation(int outK, ModelLearner modeler, RelationManager<Integer> relMngr,
			AtomicInteger crunchesLeft, ModuleDistribution<Integer> moduleDistribution) {
		Collection<TransitionMemory> experience = modeler.getExperience().getBatch(numTransitions, false);
		if (experience.isEmpty()) {
			throw new IllegalStateException("get experience please");
		}
		// store all known modules and their scores in a distribution
		recalcModuleDistribution(moduleDistribution, modeler, relMngr, outK, outK, experience);
		
		// if all the known modules suck for this output, and if there is still computation power available,
		// build a new module
		if (moduleDistribution.getHighestScore() < MIN_SCORE && crunchesLeft.getAndDecrement() > 0) {
			ReusableIntModule newModule = ReusableIntModule.createNeighborHoodModule(modeler, relMngr,
					outK, numHidden, epochs, lRate, mRate, sRate);
			allModules.add(newModule);
			double score = calcScoresFromTransition(newModule, relMngr, experience, modeler, outK, outK);
			// store the new module in the distribution
			moduleDistribution.addModule(newModule, score);
		}
		return (ReusableIntModule) moduleDistribution.getMostLikelyModule();
	}
	
	@Override
	protected double calcScoreFromTransition(ReusableModule<Integer> module, RelationManager<Integer> relMngr,
			TransitionMemory tm, ModelLearner modeler, Integer output, int outputKey) {
		final double predicted = module.evaluateOutput(output, relMngr, tm.getPreStateAndAction());
		final double observed = tm.getPostState()[outputKey];
		return 1 - Math.abs(observed - predicted);
	}
	
	private void shuffleOutK(double[] fullOutputs) {
		if (shuffledOutK == null) {
			shuffledOutK = new ArrayList<Integer>();
			for (int i = 0; i < fullOutputs.length; i++) shuffledOutK.add(i);
		}
		Collections.shuffle(shuffledOutK);
	}

	public void processFullModel(ModelLearner modeler, RelationManager<Integer> relMngr, int numTransitions) {
		loadModeler(modeler);
		this.numTransitions = numTransitions;
		AtomicInteger crunchesLeft = new AtomicInteger(maxCrunchPerRound);
		if (outputScores.isEmpty()) {
			// in case outputs have never been calculated before
			setupClean(modeler, relMngr);
		} else {
			// iterate through lowest-scoring outputs, recalculating their module distributions
			for (Iterator<OutputScore<Integer>> iter = outputScores.descendingIterator(); iter.hasNext();) {
				OutputScore<Integer> outputScore = iter.next();
				int outK = outputScore.key;
				ModuleDistribution<Integer> moduleDistribution = nodeModules.get(outK);
				if (moduleDistribution == null) nodeModules.put(outK, moduleDistribution = new ModuleDistribution<Integer>());
				learnModuleAssociation(outK, modeler, relMngr, crunchesLeft, moduleDistribution);
			}
		}
	}
	
	private void loadModeler(ModelLearner modeler) {
		if (modeler != loadedModeler && loadedModeler != null) {
			Map<Integer, ModuleDistribution<Integer>> nodeModules = new HashMap<Integer, ModuleDistribution<Integer>>();
			oldNodeModules.put(loadedModeler, new HashMap<Integer, ModuleDistribution<Integer>>());
			oldNodeModules.get(loadedModeler).putAll(nodeModules);
			oldOutputScores.put(loadedModeler, new TreeSet<OutputScore<Integer>>());
			oldOutputScores.get(loadedModeler).addAll(outputScores);
			nodeModules.clear();
			outputScores.clear();
			shuffledOutK = null;
		}
		// TODO what if you go back to using the oldModeler? you have to reload stored stuff
		loadedModeler = modeler;
	}
	
	private void setupClean(ModelLearner modeler, RelationManager<Integer> relMngr) {
		TransitionMemory tm = modeler.getExperience().getBatch(1, false).iterator().next();
		getOutputs(modeler, relMngr, tm.getPreState(), tm.getAction());
	}

	public ReusableIntModule getModuleUsedBy(int output) {
		return (ReusableIntModule) nodeModules.get(output).getMostLikelyModule();
	}

	private ReusableIntModule getOldBest(int k) {
		if (oldNodeModules.isEmpty()) return null;
		ModuleDistribution<Integer> oldBestDist = oldNodeModules.values().iterator().next().get(k);
		if (oldBestDist != null) {
			return (ReusableIntModule) oldBestDist.getMostLikelyModule();
		} else {
			return null;
		}
	}
}
