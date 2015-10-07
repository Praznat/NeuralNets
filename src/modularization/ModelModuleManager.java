package modularization;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.indirectencodings.RelationManager;
import ann.testing.GridGame;
import modeler.ModelLearner;
import modeler.TransitionMemory;

public class ModelModuleManager {

	private final GridGame game;
	private final Map<Node, ReusableModule> nodeModules = new HashMap<Node, ReusableModule>();
	private final Set<ModelLearner> modelersSeen = new HashSet<ModelLearner>();

	private double loErrorThresh = 0.1;
	private double hiErrorThresh = 0.4;

	public ModelModuleManager(GridGame game) {
		this.game = game;
	}

	/**
	 * low thresh determines how low error must be in order to create new module
	 * high thresh determines how low error must be in order to reuse old module
	 */
	public ModelModuleManager(GridGame game, double loThresh, double hiThresh) {
		this(game);
		this.loErrorThresh = loThresh;
		this.hiErrorThresh = hiThresh;
	}
	
	public double[] getOutputs(ModelLearner modeler, RelationManager relMngr, double[] stateVars, double[] action) {
		double[] inputs = Utils.concat(stateVars, action);
		FFNeuralNetwork neuralNetwork = getModelNetwork(modeler);
		FFNeuralNetwork.feedForward(neuralNetwork.getInputNodes(), inputs);
		ArrayList<? extends Node> outputNodes = neuralNetwork.getOutputNodes();
		double[] outputs = new double[stateVars.length];
		int i = 0;
		for (Node output : outputNodes) {
			ReusableModule module = nodeModules.get(output);
			// if no module use original network output
			if (module == null) outputs[i] = output.getActivation();
			// if module use module output
			else {
				outputs[i] = module.evaluateOutput(output, relMngr);
			}
			i++;
		}
		return outputs;
	}

	public void processFullModel(ModelLearner modeler, RelationManager relMngr, int numTransitions, int times) {
		for (int i = 0; i < times; i++) processFullModel(modeler, relMngr, numTransitions);
	}
	public void processFullModel(ModelLearner modeler, RelationManager relMngr, int numTransitions) {
		Collection<TransitionMemory> transitions = modeler.getExperience().getBatch(numTransitions, true);
		FFNeuralNetwork neuralNetwork = modeler.getTransitionsModule().getNeuralNetwork();
		ArrayList<? extends Node> outputNodes = neuralNetwork.getOutputNodes();
		if (!modelersSeen.contains(modeler)) {
			int s = modelersSeen.size();
			int i = 0;
			for (Node n : outputNodes) n.setName(buffString(s + i++/100.0 + "", 4));
			modelersSeen.add(modeler);
		}
		double[] errors = new double[outputNodes.size()];
		for (TransitionMemory tm : transitions) {
			FFNeuralNetwork.feedForward(neuralNetwork.getInputNodes(), tm.getPreStateAndAction());
			double[] observed = tm.getPostState();
			int i = 0;
			for (Node output : outputNodes) {
				ReusableModule module = nodeModules.get(output);
				// if no module use original network error
				if (module == null) errors[i] += Math.abs(output.getActivation() - observed[i]);
				// if module use module error
				else {
					errors[i] = calcErrorFromTransition(module, relMngr, tm, modeler, output, i);
				}
				i++;
			}
		}
		int n = transitions.size();
		for (int i = 0; i < errors.length; i++) errors[i] /= n;
		respondToErrors(errors, modeler, relMngr, outputNodes, transitions);
	}
	
	private void respondToErrors(double[] errors, ModelLearner modeler, RelationManager relMngr,
			ArrayList<? extends Node> outputs, Collection<TransitionMemory> transitions) {
		for (int i = 0; i < errors.length; i++) {
			final double error = errors[i];
			Node output = outputs.get(i);
			if (error < loErrorThresh) {
				// dont worry if it already has module
				if (nodeModules.get(output) != null) continue;
				// pick best existing module
				ReusableModule bestModule = pickModule(modeler, relMngr, output, i, transitions, loErrorThresh);
				if (bestModule != null) nodeModules.put(output, bestModule);
				// create new module if no existing ones work
				else {
					ReusableModule module = ReusableModule.createNeighborHoodModule(game, modeler, relMngr,
							getModelNetwork(modeler), output);
					nodeModules.put(output, module);
				}
			} else if (error > hiErrorThresh) {
				// pick best module if lower than current error
				ReusableModule bestModule = pickModule(modeler, relMngr, output, i, transitions, error);
				if (bestModule != null) nodeModules.put(output, bestModule);
			}
		}
	}

	public ReusableModule pickModule(ModelLearner modeler, RelationManager relMngr, Node output, int outputKey,
			int numTransitions, double errorThresh) {
		Collection<TransitionMemory> transitions = modeler.getExperience().getBatch(numTransitions, true);
		return pickModule(modeler, relMngr, output, outputKey, transitions, errorThresh);
	}
	
	public ReusableModule pickModule(ModelLearner modeler, RelationManager relMngr, Node output, int outputKey,
			Collection<TransitionMemory> transitions, double errorThresh) {
		double lowestError = errorThresh;
		ReusableModule best = null;
		for (ReusableModule module : nodeModules.values()) {
			double meanerror = 0;
			for (TransitionMemory tm : transitions) {
				meanerror += calcErrorFromTransition(module, relMngr, tm, modeler, output, outputKey);
			}
			meanerror /= transitions.size();
			if (meanerror < lowestError) {
				lowestError = meanerror;
				best = module;
			}
		}
		return best;
	}
	
	private double calcErrorFromTransition(ReusableModule module, RelationManager relMngr, TransitionMemory tm,
			ModelLearner modeler, Node output, int outputKey) {
		double[] input = tm.getPreStateAndAction();
		
		ArrayList<? extends Node> inputNodes = getModelNetwork(modeler).getInputNodes();
		int i = 0;
		for (Node n : inputNodes) n.clamp(input[i++]);
		
		final double predicted = module.evaluateOutput(output, relMngr);
		final double observed = tm.getPostState()[outputKey];
		return Math.abs(observed - predicted);
	}
	
	private static FFNeuralNetwork getModelNetwork(ModelLearner model) {
		return model.getTransitionsModule().getNeuralNetwork();
	}
	
	private static String buffString(String s, int len) {
		int r = len - s.length();
		for (int i = 0; i < r; i++) s = s.toString() + "0";
		return s.substring(0, len);
	}

	public void report() {
		System.out.println("Reporting on Module Manager");
		Set<ReusableModule> allModules = new HashSet<ReusableModule>();
		allModules.addAll(nodeModules.values());
		System.out.println(allModules.size() + " modules total");
		TreeSet<Node> ordered = new TreeSet<Node>(new Comparator<Node>() {
			@Override
			public int compare(Node arg0, Node arg1) {
				return arg0.toString().compareTo(arg1.toString());
			}
		});
		ordered.addAll(nodeModules.keySet());
		for (Node n : ordered) {
			System.out.println(n + "	" + nodeModules.get(n));
		}
		
	}
}
