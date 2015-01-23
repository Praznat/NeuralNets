package deepnets.testing;

import java.util.*;

import modeler.*;

import reasoner.Foresight;
import utils.RandomUtils;

import deepnets.*;

public class BasicTests {

	public static void main(String[] args) {
		xor();
		bayesian();
		envtranslator();
		transitionApproximator();
//		modeling();
		fakePole();
	}

	private static void xor() {
		System.out.println("XOR");
		Layer<Node> inputLayer = Layer.createInputLayer(2);
		Layer<Node> hiddenLayer = Layer.createHiddenFromInputLayer(inputLayer, 2, ActivationFunction.SUPERSIGMOID);
		BiasNode.connectToLayer(hiddenLayer);
		Layer<Node> outputLayer = Layer.createHiddenFromInputLayer(hiddenLayer, 1, ActivationFunction.SUPERSIGMOID);
		BiasNode.connectToLayer(outputLayer);
		double[][] inputSamples = {{0,0},{0,1},{1,0},{1,1}};
		System.out.println("Untrained");
		for (double[] is : inputSamples) {
			FFNeuralNetwork.feedForward(inputLayer.getNodes(), is);
			String output = "";
			for (Node n : outputLayer.getNodes()) output += n.getActivation() + " ";
			System.out.println(output);
		}
		
		double[][] outputSamples = {{0},{1},{1},{0}};
		
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);
		System.out.println("E: " + FFNeuralNetwork.stdError(inputLayer.getNodes(), outputLayer.getNodes(), data));
		
		ControlPanel.learnFromBackPropagation(inputLayer.getNodes(), outputLayer.getNodes(), data, 10000,
				2, 0.8, 0.5, 0.3, 0.05, 0.01);

		System.out.println("Trained");
		for (double[] is : inputSamples) {
			FFNeuralNetwork.feedForward(inputLayer.getNodes(), is);
			String output = "";
			for (Node n : outputLayer.getNodes()) output += n.getActivation() + " ";
			System.out.println(output);
		}
		System.out.println("E: " + FFNeuralNetwork.stdError(inputLayer.getNodes(), outputLayer.getNodes(), data));
		
	}

	/**
	 * test to see if neural net can represent bayesian transitions
	 * @return
	 */
	private static boolean bayesian() {
		System.out.println("BAYESIAN");
		
//		FFNeuralNetwork ffnn = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5,5,5);
		VariableTransitionApproximator modeler = new VariableTransitionApproximator(100, new int[] {}, ActivationFunction.SIGMOID0p5, 10);
		
		Collection<DataPoint> data = new ArrayList<DataPoint>();
		data.add(new DataPoint(new double[] {0,0,1,0,0}, new double[] {0,0,0,1,0})); // move right
		data.add(new DataPoint(new double[] {0,0,1,0,0}, new double[] {0,1,0,0,0})); // move left
		data.add(new DataPoint(new double[] {0,1,0,0,0}, new double[] {1,0,0,0,0})); // move left again
		data.add(new DataPoint(new double[] {0,0,0,1,0}, new double[] {0,0,1,0,0})); // move back to center
		data.add(new DataPoint(new double[] {0,0,0,1,0}, new double[] {0,0,0,0,0})); // disappear
		
//		ControlPanel.learnFromBackPropagation(ffnn.getInputNodes(), ffnn.getOutputNodes(), data,
//				10000, 1,1,0,0,0,0);
		for (DataPoint dp : data) {
			modeler.observePreState(dp.getInputs());
			modeler.observePostState(dp.getOutputs());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5,0.5,0, false, 1000);
		modeler.getNeuralNetwork().report(data);
		
		double[] foresight = Foresight.recurse(modeler.getNeuralNetwork(), new double[] {0,0,1,0,0}, 1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(foresight[0] < 0.1 && foresight[1] > 0.4 && foresight[2] < 0.1 && foresight[3] > 0.4 && foresight[4] < 0.1
			? "recurse 1 ok" : "recurse 1 sucks");
		foresight = Foresight.recurse(modeler.getNeuralNetwork(), new double[] {0,0,1,0,0}, 2);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(foresight[0] > 0.4 && foresight[1] < 0.1 && foresight[2] > 0.2 && foresight[3] < 0.1 && foresight[4] < 0.1
				? "recurse 2 ok" : "recurse 2 sucks");
		
		foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, 1, 100000, 0.1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(foresight[0] < 0.1 && foresight[1] > 0.4 && foresight[2] < 0.1 && foresight[3] > 0.4 && foresight[4] < 0.1
			? "montecarlo 1 ok" : "montecarlo 1 sucks");
		foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, 2, 100000, 0.1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(foresight[0] > 0.4 && foresight[1] < 0.12 && foresight[2] > 0.2 && foresight[3] < 0.12 && foresight[4] < 0.12
				? "montecarlo 2 ok" : "montecarlo 2 sucks");
		
		return false;
	}
	

	private static void envtranslator() {
		final EnvTranslator bucketer = EnvTranslator.rbfEnvTranslator(
				new double[] {0, 0, -2.5}, new double[] {1.0, 5.0, 2.5}, new int[] {5, 5, 3}, 0.8);
		double[] testVals = {.33, 5.0, -0.2};
		double[] ins = bucketer.toNN(testVals);
		double[] back = bucketer.fromNN(ins);
		String s1 = "", s2 = "";
		for (double d : ins) s1 += d + "	";
		for (double d : back) s2 += d + "	";
		System.out.println("INS	" + s1);
		System.out.println("BACK	" + s2);
	}
	
	private static void transitionApproximator() {
		VariableTransitionApproximator modeler = new VariableTransitionApproximator(500,
				new int[] {5}, ActivationFunction.SIGMOID0p5, 50);
		double[][] inputSamples = {{0,0},{0,1},{1,0},{1,1}};
		double[][] outputSamples = {{0},{1},{1},{0}};
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);

		for (DataPoint dp : data) {
			double[] inputs = dp.getInputs();
			modeler.observePreState(inputs[0]);
			modeler.observeAction(inputs[1]);
			modeler.observePostState(dp.getOutputs());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(2, 0.5, 0, false, 1000);
		modeler.getNeuralNetwork().report(data);
	}

	
	private static void fakePole() {
		//TODO pls try to get this working with more buckets
		VariableTransitionApproximator modeler = new VariableTransitionApproximator(100,
				new int[] {30}, ActivationFunction.SIGMOID0p5, 500);

		final boolean NN_FORM = false;
		double[] mins = Pole.stateMins;
		double[] maxes = Pole.stateMaxes;
		EnvTranslator stateTranslator = EnvTranslator.rbfEnvTranslator(mins, maxes, new int[] {12,12}, .5);
		EnvTranslator actTranslator = Pole.actionTranslator;
		List<double[]> actions = Pole.actionChoices;
		actions.add(new double[] {1,0});
//		actions.add(new double[] {0,0});
		actions.add(new double[] {0,1});
		final Collection<double[]> tmpCorrel = new ArrayList<double[]>();
		for (int t = 0; t < 2000; t++) {
			double[] preState = new double[mins.length];
			for (int i = 0; i < preState.length; i++) {
				final double spread = (maxes[i] - mins[i]) / 10;
				preState[i] = RandomUtils.randBetween(mins[i] - spread, maxes[i] + spread);	
			}
			double[] inNN = stateTranslator.toNN(preState);
			double[] action = RandomUtils.randomOf(actions);
			modeler.observePreState(inNN);
			modeler.observeAction(action);
			
			double[] postState = new double[mins.length];
			double act = Math.random() < 0.99 ? action[0] - action[1] : (2*Math.round(Math.random())-1);
			double r = act/100;
			for (int i = 0; i < postState.length; i++) {
				postState[i] = preState[i] * Math.exp(Math.signum(preState[i]) * (i == 0 ? r : -r));
			}
			tmpCorrel.add(new double[] {act, postState[0] - preState[0]});
			modeler.observePostState(stateTranslator.toNN(postState));
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 1000, 0.0003); // IT SEEMS CONFIDENCE IS NOT RELIABLE INDICATOR
//		for (double[] dd : tmpCorrel) {
//			String s = ""; for (double d : dd) s += d + "	";
//			System.out.println(s);
//		}
		for (int i = 0; i < 10; i++) System.out.println("*********");
		System.out.println(modeler.getConfidenceEstimate());
		modeler.testit(1000, mins, maxes, stateTranslator, actTranslator, actions, NN_FORM);
	}

	/** THIS IS WORKING CORRECTLY */
	private static void modeling() {
		EnvTranslator stateTranslator = new EnvTranslator() {
			double[][] dd = {{1,0,0},{0,1,0},{0,0,1}};
			@Override
			public double[] toNN(double... n) {
				return dd[(int) n[0]];
			}
			@Override
			public double[] fromNN(double[] d) {
				return new double[] {d[0] * 0 + d[1] * 1 + d[2] * 2};
			}
		};
		EnvTranslator actTranslator = new EnvTranslator() {
			public double[] toNN(double... n) { return n; }
			public double[] fromNN(double[] d) { return d; }
		};
		VariableTransitionApproximator modeler = new VariableTransitionApproximator(100,
				new int[] {5}, ActivationFunction.SIGMOID0p5, 500);
		double[] envinputs = {0,1,2};
		List<double[]> actions = Pole.actionChoices;
		actions.add(new double[] {0});
		actions.add(new double[] {1});
		for (int t = 0; t < 2000; t++) {
			double envinput = RandomUtils.randomOf(envinputs);
			double[] action = RandomUtils.randomOf(actions);
			double envoutput = ((action[0] == 0 ? envinput - 1 : envinput + 1) + envinputs.length) % envinputs.length;
			double[] nnIn = stateTranslator.toNN(envinput);
			double[] nnOut = stateTranslator.toNN(envoutput);
			modeler.observePreState(nnIn);
			modeler.observeAction(action);
			modeler.observePostState(nnOut);
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 5000);
		for (int i = 0; i < 10; i++) System.out.println("*********");
		System.out.println(modeler.getConfidenceEstimate());
		modeler.testit(1000, new double[] {envinputs[0]}, new double[] {envinputs[envinputs.length-1]},
				stateTranslator, actTranslator, actions, true);
		System.out.println("????????");
	}
	
}
