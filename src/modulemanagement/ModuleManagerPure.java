package modulemanagement;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeSet;

import ann.Utils;
import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modeler.TransitionMemory;
import reasoner.DiscreteState;
import reasoner.Foresight;

public class ModuleManagerPure extends ModuleManager<Integer> {
	
	private double minScore;
	private int maxCrunchPerRound;
	private double killPct;
	private final int[] numHidden;
	private final int nnTrainingEpochs;
	private final double lRate;
	private final double mRate;
	private final double sRate;
	private Map<ModelLearner, TreeSet<OutputScore<Integer>>> oldOutputScores
		= new HashMap<ModelLearner, TreeSet<OutputScore<Integer>>>();
	private Map<ModelLearner, Map<Integer, ModuleDistribution<Integer>>> oldNodeModules
		= new HashMap<ModelLearner, Map<Integer, ModuleDistribution<Integer>>>();
	
	private List<Integer> shuffledOutK = new ArrayList<Integer>();
	private int numTransitions;
	private ModelLearner loadedModeler;
	
	/** maxCrunchPerRound specifies how many new modules you can learn in a round */
	// TODO have max number of modules?
	public ModuleManagerPure(double minScore, int maxCrunchPerRound, double killPct,
			double lRate, double mRate, double sRate, int[] numHidden, int nnTrainingEpochs) {
		this.minScore = minScore;
		this.maxCrunchPerRound = maxCrunchPerRound;
		this.killPct = killPct;
		this.numHidden = numHidden;
		this.nnTrainingEpochs = nnTrainingEpochs;
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
//		outputScores.clear(); // store them later in this method

		// look through output vars in random order
		// get prediction for each one
		for (int outK : shuffledOutK) {
			ModuleDistribution<Integer> moduleDistribution = nodeModules.get(outK);
			if (moduleDistribution == null) nodeModules.put(outK, moduleDistribution = new ModuleDistribution<Integer>());
			fullOutputs[outK] = getOutput(outK, moduleDistribution, fullInputs, modeler, relMngr);
		}
		return fullOutputs;
	}

	/** returns a predicted output */
	private double getOutput(int outK, ModuleDistribution<Integer> moduleDistribution, double[] fullInputs,
			ModelLearner modeler, RelationManager<Integer> relMngr) {
		ReusableIntModule mostLikelyModule = (ReusableIntModule) moduleDistribution.getMostLikelyModule();
		// if no modules in distribution for this output, have to learn association ... (what about score?)
		if (mostLikelyModule == null) {
//			System.out.println("Calculating module scores for output " + outK + ". Assigned outputs: " + nodeModules.size());
			mostLikelyModule = learnModuleAssociation(outK, modeler, relMngr, moduleDistribution);
		}
		// once you have a module, make the prediction
		double predOut = mostLikelyModule.evaluateOutput(outK, relMngr, fullInputs);
		predOut = Math.random() < Foresight.probabilitySkewing(predOut, 0.08) ? 1 : 0;
		// remember the score for this output
//		double score = moduleDistribution.getHighestScore();
//		addOutputScore(outK, score, outK);
//		System.out.println(outK + "	" + score + "	" + moduleDistribution.size());
		return predOut;
	}

	private ReusableIntModule learnModuleAssociation(int outK, ModelLearner modeler,
			RelationManager<Integer> relMngr, ModuleDistribution<Integer> moduleDistribution) {
		Collection<TransitionMemory> experience = modeler.getExperience().getBatch(numTransitions, false);
		if (experience.isEmpty()) {
			throw new IllegalStateException("get experience please");
		}
		// store all known modules and their scores in a distribution
		recalcModuleDistribution(moduleDistribution, modeler, relMngr, outK, outK, experience);
		
		// if all the known modules suck for this output, and if there is still computation power available,
		// build a new module
		if (moduleDistribution.getHighestScore() < minScore) {
			if (allModules.size() >= maxCrunchPerRound) cleanupTheCrappy(killPct);
			if (allModules.size() < maxCrunchPerRound) { // not "else"... cleanup might reduce allModules size
				System.out.println("Creating new module " + allModules.size() + " for output " + outK
						+ " ... already associated " + nodeModules.size());
				ReusableIntModule newModule = ReusableIntModule.createNeighborHoodModule(relMngr);
//			ReusableIntModule newModule = ReusableIntModule.createNeighborHoodModule(modeler, relMngr,
//					outK, numHidden, nnTrainingEpochs, lRate, mRate, sRate);
				allModules.add(newModule);
				double score = calcScoresFromTransition(moduleDistribution, newModule, relMngr, experience, modeler, outK, outK);
				// store the new module in the distribution
				moduleDistribution.addModule(newModule, score);
			}
		}
		return (ReusableIntModule) moduleDistribution.getMostLikelyModule();
	}
	
	@Override
	protected void trainModule(ReusableModule<Integer> module, Map<DiscreteState, Double> out1f) {
		module.trainIfNecessary(out1f, numHidden, nnTrainingEpochs, lRate, mRate, sRate);
	}
	
	@Override
	protected double calcScoreFromTransition(ReusableModule<Integer> module, RelationManager<Integer> relMngr,
			TransitionMemory tm, ModelLearner modeler, Integer output, int outputKey) {
		final double predicted = module.evaluateOutput(output, relMngr, tm.getPreStateAndAction());
		final double observed = tm.getPostState()[outputKey];
		return 1 - Math.abs(observed - predicted);
	}
	
	private void shuffleOutK(double[] fullOutputs) {
		if (shuffledOutK.size() < fullOutputs.length) {
			shuffledOutK.clear();
			for (int i = 0; i < fullOutputs.length; i++) shuffledOutK.add(i);
		}
		Collections.shuffle(shuffledOutK);
	}

	public void processFullModel(ModelLearner modeler, RelationManager<Integer> relMngr, int numTransitions) {
		System.out.println("Processing model. Num modules: " + allModules.size());
		loadModeler(modeler);
		this.numTransitions = numTransitions;
		Collection<OutputScore<Integer>> outputScorez = getDescOutputScores();
		if (outputScorez.isEmpty()) {
			// in case outputs have never been calculated before
			setupClean(modeler, relMngr);
		} else {
			long ms = System.currentTimeMillis();
			int i = 0;
			// iterate through lowest-scoring outputs, recalculating their module distributions
			for (OutputScore<Integer> outputScore : outputScorez) {
				if (i < 30) System.out.println(i + "th worst is " + outputScore.key);
				long nowMs = System.currentTimeMillis();
				if (nowMs - ms > 10000) {
					System.out.println("Done processing " + i + " outputs out of " + outputScorez.size());
					ms = nowMs;
				}
				int outK = outputScore.key;
				ModuleDistribution<Integer> moduleDistribution = nodeModules.get(outK);
				if (moduleDistribution == null) nodeModules.put(outK, moduleDistribution = new ModuleDistribution<Integer>());
				learnModuleAssociation(outK, modeler, relMngr, moduleDistribution);
				i++;
			}
		}
	}

	@Override
	protected Collection<OutputScore<Integer>> getDescOutputScores() {
		TreeSet<OutputScore<Integer>> outScores = new TreeSet<OutputScore<Integer>>();
		for (Entry<Integer, ModuleDistribution<Integer>> entry : nodeModules.entrySet()) {
			int k = entry.getKey();
			ModuleDistribution<Integer> dist = entry.getValue();
			double score = dist.getHighestScore() + (Math.random()-.5)/1000000;
			outScores.add(new OutputScore<Integer>(k, score, k));
		}
		return outScores;
	}
	
	protected void cleanupTheCrappy(double killPct) {
		Map<ReusableModule<Integer>, Double> needsMap = getNeeds();
		ArrayList<Double> scores = new ArrayList<Double>(needsMap.values());
		Collections.sort(scores); // lowest to highest
		int killIndex = (int) Math.round(killPct * scores.size());
		double minScore = scores.get(killIndex);
		int maxKills = (int) (1 / minScore);
		if (minScore >= 0.5) return;
		int kills = 0;
		for (Entry<ReusableModule<Integer>, Double> entry : needsMap.entrySet()) {
			double score = entry.getValue();
			if (score < minScore && kills++ < maxKills) {
				System.out.println("PURGING MODULE WITH SCORE	" + score);
				purgeModule(entry.getKey());
			}
		}
	}
	
	protected void purgeModule(ReusableModule<Integer> module) {
		allModules.remove(module);
		for (ModuleDistribution<Integer> dist : nodeModules.values()) {
			dist.removeModule(module);
		}
	}
	
	protected Map<ReusableModule<Integer>, Double> getNeeds() {
		Map<ReusableModule<Integer>, Double> needs = new HashMap<ReusableModule<Integer>, Double>();
		for (ModuleDistribution<Integer> dist : nodeModules.values()) {
			Collection<ReusableModule<Integer>> goods = dist.getModulesAboveThresh(minScore);
			if (goods.isEmpty()) continue;
			for (ReusableModule<Integer> module : allModules) {
				double presence = goods.contains(module) ? 1 : 0;
				double need = presence / goods.size();
				Double oldNeed = needs.get(module);
				if (oldNeed == null || need > oldNeed) needs.put(module, need);
			}
		}
		return needs;
	}

	@Override
	protected void reportModulePopularity() {
		Map<ReusableModule<Integer>, Double> needs = getNeeds();
		for (Entry<ReusableModule<Integer>, Double> entry : needs.entrySet()) {
			System.out.println(entry.getValue() + "	" + entry.getKey());
		}
	}
	
	public void setMaxModules(int m) {
		maxCrunchPerRound = m;
	}
	
	public void setMinScore(double s) {
		minScore = s;
	}
	
	private void loadModeler(ModelLearner modeler) {
		if (modeler != loadedModeler && loadedModeler != null) {
			for (ModuleDistribution<Integer> dist : nodeModules.values()) {
				dist.getData().clear();
			}
			oldNodeModules.put(loadedModeler, new HashMap<Integer, ModuleDistribution<Integer>>());
			oldNodeModules.get(loadedModeler).putAll(nodeModules);
			oldOutputScores.put(loadedModeler, new TreeSet<OutputScore<Integer>>());
//			oldOutputScores.get(loadedModeler).addAll(outputScores);
			nodeModules.clear();
//			outputScores.clear();
			shuffledOutK.clear();
		}
		// TODO what if you go back to using the oldModeler? you have to reload stored stuff
		// TODO get rid of this transfer stuff
		loadedModeler = modeler;
	}
	
	private void setupClean(ModelLearner modeler, RelationManager<Integer> relMngr) {
		TransitionMemory tm = modeler.getExperience().getBatch(1, false).iterator().next();
		getOutputs(modeler, relMngr, tm.getPreState(), tm.getAction());
	}

	public ReusableIntModule getModuleUsedBy(int output) {
		return (ReusableIntModule) nodeModules.get(output).getMostLikelyModule();
	}

	@SuppressWarnings("unused")
	private ReusableIntModule getOldBest(int k) { // for debugging
		if (oldNodeModules.isEmpty()) return null;
		ModuleDistribution<Integer> oldBestDist = oldNodeModules.values().iterator().next().get(k);
		if (oldBestDist != null) {
			return (ReusableIntModule) oldBestDist.getMostLikelyModule();
		} else {
			return null;
		}
	}
	
	public static void saveState(ModuleManagerPure mmp, String namey) {
		SaveState ss = new SaveState(mmp.nodeModules, mmp.allModules, //mmp.outputScores,
				mmp.minScore, mmp.maxCrunchPerRound, mmp.killPct,
				mmp.numHidden, mmp.nnTrainingEpochs, mmp.lRate, mmp.mRate, mmp.sRate);
		Utils.saveObjectToFile(namey, ss);
	}
	
	public static ModuleManagerPure loadState(String namey) {
		SaveState ss = (SaveState) Utils.loadObjectFromFile(namey);
		ModuleManagerPure result = new ModuleManagerPure(ss.minScore, ss.maxCrunch, ss.killPct,
				ss.lRate, ss.mRate, ss.sRate, ss.numHidden, ss.nnTrainingEpochs);
		result.nodeModules.putAll(ss.nodeModules);
		result.allModules.addAll(ss.allModules);
//		result.outputScores.addAll(ss.outputScores);
		return result;
	}
	
	@SuppressWarnings("serial")
	private static class SaveState implements Serializable {
		private Map<Integer, ModuleDistribution<Integer>> nodeModules;
		private Collection<ReusableModule<Integer>> allModules;
//		private TreeSet<OutputScore<Integer>> outputScores;
		private double minScore;
		private int maxCrunch;
		private double killPct;
		private int[] numHidden;
		private double lRate;
		private int nnTrainingEpochs;
		private double mRate;
		private double sRate;

		private SaveState(Map<Integer, ModuleDistribution<Integer>> nodeModules,
				Collection<ReusableModule<Integer>> allModules, //TreeSet<OutputScore<Integer>> outputScores,
				double minScore, int maxCrunch, double killPct, int[] numHidden, int nnTrainingEpochs,
				double lRate, double mRate, double sRate) {
					this.nodeModules = nodeModules;
					this.allModules = allModules;
//					this.outputScores = outputScores;
					this.minScore = minScore;
					this.maxCrunch = maxCrunch;
					this.killPct = killPct;
					this.numHidden = numHidden;
					this.nnTrainingEpochs = nnTrainingEpochs;
					this.lRate = lRate;
					this.mRate = mRate;
					this.sRate = sRate;
		}
	}
}
