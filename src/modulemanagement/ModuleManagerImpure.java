package modulemanagement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modeler.TransitionMemory;

public class ModuleManagerImpure extends ModuleManager<Node> {

	private final Set<ModelLearner> modelersSeen = new HashSet<ModelLearner>();
	
	private double hiScoreThresh = 0.9;
	private double loScoreThresh = 0.6;

	/**
	 * high thresh determines how high score must be in order to create new module
	 * low thresh determines how high score must be in order to reuse old module
	 */
	public ModuleManagerImpure(double hiThresh, double loThresh) {
		this.hiScoreThresh = hiThresh;
		this.loScoreThresh = loThresh;
	}
	
	/** return output activations given state and action inputs */
	@Override
	public double[] getOutputs(ModelLearner modeler, RelationManager<Node> relMngr, double[] stateVars, double[] action) {
		double[] inputs = Utils.concat(stateVars, action);
		FFNeuralNetwork neuralNetwork = getModelNetwork(modeler);
		FFNeuralNetwork.feedForward(neuralNetwork.getInputNodes(), inputs);
		ArrayList<? extends Node> outputNodes = neuralNetwork.getOutputNodes();
		double[] outputs = new double[stateVars.length];
		int i = 0;
		for (Node output : outputNodes) {
			ModuleDistribution<Node> moduleDist = nodeModules.get(output);
			// if no module use original network output
			if (moduleDist == null) {
				outputs[i] = output.getActivation();
			}
			// if module use module output
			else {
				outputs[i] = moduleDist.getMostLikelyModule().evaluateOutput(output, relMngr, null);
			}
			i++;
		}
		return outputs;
	}

	@Override
	public void processFullModel(ModelLearner modeler, RelationManager<Node> relMngr, int numTransitions) {
		Collection<TransitionMemory> transitions = modeler.getExperience().getBatch(numTransitions, false);
		FFNeuralNetwork neuralNetwork = modeler.getTransitionsModule().getNeuralNetwork();
		ArrayList<? extends Node> outputNodes = neuralNetwork.getOutputNodes();
		if (!modelersSeen.contains(modeler)) {
			int s = modelersSeen.size();
			int i = 0;
			for (Node n : outputNodes) n.setName(buffString(s + i++/100.0 + "", 4));
			modelersSeen.add(modeler);
		}
		double[] scores = new double[outputNodes.size()];
		for (TransitionMemory tm : transitions) {
			FFNeuralNetwork.feedForward(neuralNetwork.getInputNodes(), tm.getPreStateAndAction());
			double[] observed = tm.getPostState();
			int i = 0;
			for (Node output : outputNodes) {
				ModuleDistribution<Node> moduleDist = nodeModules.get(output);
				// if no module use original network score
				if (moduleDist == null) scores[i] += (1 - Math.abs(output.getActivation() - observed[i]));
				// if module use module score
				else {
					scores[i] += calcScoreFromTransition(moduleDist.getMostLikelyModule(), relMngr, tm, modeler, output, i);
				}
				i++;
			}
		}
		int n = transitions.size();
//		outputScores.clear();
//		for (int i = 0; i < scores.length; i++) {
//			addOutputScore(outputNodes.get(i), scores[i] / n, i);
//		}
//		respond(outputScores, modeler, relMngr, transitions);
	}

	private void respond(TreeSet<OutputScore<Node>> outputScores, ModelLearner modeler,
			RelationManager<Node> relMngr, Collection<TransitionMemory> transitions) {
		// has to be in descending score order!
		// im not really sure why but if it's ascending you create way more modules than you really want
		for (Iterator<OutputScore<Node>> iter = outputScores.descendingIterator(); iter.hasNext();) {
			OutputScore<Node> outputScore = iter.next();
			final double score = outputScore.score;
			final Node output = outputScore.output;
			final int i = outputScore.key;
//			System.out.println(output + "	" + error);
			if (score > loScoreThresh) {
				// dont worry if it already has module
				if (nodeModules.get(output) != null) continue;
				// pick best existing module
				ModuleDistribution<Node> moduleDist = new ModuleDistribution<Node>();
				recalcModuleDistribution(moduleDist, modeler, relMngr, output, i, transitions);
				if (!moduleDist.isEmpty() && moduleDist.getHighestScore() > score) nodeModules.put(output, moduleDist);
				// create new module if no existing ones work
				else {
//					System.out.println("new module created for " + output);
					ReusableNNModule module = ReusableNNModule.createNeighborHoodModule(modeler, relMngr,
							getModelNetwork(modeler), output);
					allModules.add(module);
					nodeModules.put(output, new ModuleDistribution<Node>(module, score));
				}
			} else if (score > hiScoreThresh) {
				// pick best module if lower than current error
				ModuleDistribution<Node> moduleDist = new ModuleDistribution<Node>();
				recalcModuleDistribution(moduleDist, modeler, relMngr, output, i, transitions);
				if (!moduleDist.isEmpty() && moduleDist.getHighestScore() > score) nodeModules.put(output, moduleDist);
			}
		}
	}
	
	public ReusableNNModule getModuleUsedBy(Node output) {
		return (ReusableNNModule) nodeModules.get(output).getMostLikelyModule();
	}
	
	@Override
	protected double calcScoreFromTransition(ReusableModule<Node> module, RelationManager<Node> relMngr,
			TransitionMemory tm, ModelLearner modeler, Node output, int outputKey) {
		double[] input = tm.getPreStateAndAction();
		
		ArrayList<? extends Node> inputNodes = getModelNetwork(modeler).getInputNodes();
		int i = 0;
		for (Node n : inputNodes) n.clamp(input[i++]);
		
		final double predicted = module.evaluateOutput(output, relMngr, null);
		final double observed = tm.getPostState()[outputKey];
		return 1 - Math.abs(observed - predicted);
	}
	
	private static FFNeuralNetwork getModelNetwork(ModelLearner model) {
		return model.getTransitionsModule().getNeuralNetwork();
	}
	
	private static String buffString(String s, int len) {
		int r = len - s.length();
		for (int i = 0; i < r; i++) s = s.toString() + "0";
		return s.substring(0, len);
	}
}
