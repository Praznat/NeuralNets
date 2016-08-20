package modulemanagement;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;

import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modeler.TransitionMemory;
import reasoner.DiscreteState;

public abstract class ModuleManager<T> {

	protected final Map<T, ModuleDistribution<T>> nodeModules = new HashMap<T, ModuleDistribution<T>>();
	protected final Collection<ReusableModule<T>> allModules = new HashSet<ReusableModule<T>>();
//	protected final TreeSet<OutputScore<T>> outputScores = new TreeSet<OutputScore<T>>();
	
	public abstract double[] getOutputs(ModelLearner modeler, RelationManager<T> relMngr, double[] stateVars, double[] action);
	public abstract void processFullModel(ModelLearner modeler, RelationManager<T> relMngr, int numTransitions);
	public void processFullModel(ModelLearner modeler, RelationManager<T> relMngr, int numTransitions, int times) {
		long ms = System.currentTimeMillis();
		long startMs = ms;
		for (int i = 0; i < times; i++) {
			processFullModel(modeler, relMngr, numTransitions);
			long nowMs = System.currentTimeMillis();
			System.out.println("Processed once in " + (nowMs - ms) + " milliseconds.");
			ms = nowMs;
		}
		System.out.println("Done processing: total time took: " + (System.currentTimeMillis() - startMs));
		reportModulePopularity();
	}
	
	protected void reportModulePopularity() {}
	
	public Map<T, ModuleDistribution<T>> getNodeModules() {
		return nodeModules;
	}
	
	protected ModuleDistribution<T> recalcModuleDistribution(ModuleDistribution<T> result, ModelLearner modeler,
			RelationManager<T> relMngr, T output, int outputKey, Collection<TransitionMemory> transitions) {
		if (allModules.isEmpty()) return result;
		result.clear();
		for (ReusableModule<T> module : allModules) {
			double score = calcScoresFromTransition(result, module, relMngr, transitions, modeler, output, outputKey);
			result.addModule(module, score);
		}
		return result;
	}
	protected double calcScoresFromTransition(ModuleDistribution<T> result, ReusableModule<T> module, RelationManager<T> relMngr,
			Collection<TransitionMemory> transitions, ModelLearner modeler, T output, int outputKey) {
		Map<DiscreteState, Double> out1f = checkLocalFrequencyMap(result, module, relMngr, transitions, output, outputKey);
		double sum = 0;
		// average error over every observed possible input
		for (Map.Entry<DiscreteState, Double> entry : out1f.entrySet()) {
			double nnOutput = module.getNNOutput(entry.getKey().getRawState());
			double target = entry.getValue();
			double diff = nnOutput - target;
			sum += 1 - Math.abs(diff); // try power
		}
		return sum / out1f.size();
	}
	protected Map<DiscreteState, Double> checkLocalFrequencyMap(ModuleDistribution<T> dist,
			ReusableModule<T> module, RelationManager<T> relMngr,
			Collection<TransitionMemory> transitions, T output, int outputKey) {
		Map<DiscreteState, Double> out1f = dist.getData(); //out1Frequencies.get(output);
		if (out1f.isEmpty()) {
			//		out1Frequencies.put(output, out1f = new HashMap<DiscreteState, Double>());
			Map<DiscreteState, AtomicInteger> countPerInput = new HashMap<DiscreteState, AtomicInteger>();
			Map<DiscreteState, AtomicInteger> sum1PerInput = new HashMap<DiscreteState, AtomicInteger>();
			for (TransitionMemory tm : transitions) {
				double[] inputs = module.getInputs(output, relMngr, tm.getPreStateAndAction());
				double outAct = tm.getPostState()[outputKey];
				//			if (outputKey >= 1728 && outputKey < 1760 && tm.getPostState()[outputKey] > 0) {
				////					&& ((inputs[0]>0&&inputs[inputs.length-2]>0) || (inputs[1]>0&&inputs[inputs.length-4]>0))) {
				//				System.out.println();
				//			}
				DiscreteState input = new DiscreteState(inputs);
				AtomicInteger count = countPerInput.get(input);
				if (count == null) countPerInput.put(input, count = new AtomicInteger());
				count.incrementAndGet();
				AtomicInteger sum = sum1PerInput.get(input);
				if (sum == null) sum1PerInput.put(input, sum = new AtomicInteger());
				if (outAct > 0) sum.incrementAndGet();
			}
			for (DiscreteState ds : countPerInput.keySet()) {
				double frequency = sum1PerInput.get(ds).doubleValue() / countPerInput.get(ds).doubleValue();
				out1f.put(ds, frequency);
			}
		}
		trainModule(module, out1f);
		return out1f;
	}
	protected void trainModule(ReusableModule<T> module, Map<DiscreteState, Double> out1f) {
		throw new IllegalStateException("Seriously you should reduce everything to PURE only");
	}
	protected double calcScoresFromTransitionOLD(ReusableModule<T> module, RelationManager<T> relMngr,
			Collection<TransitionMemory> transitions, ModelLearner modeler, T output, int outputKey) {
		double score = 0;
		for (TransitionMemory tm : transitions) {
			score += calcScoreFromTransition(module, relMngr, tm, modeler, output, outputKey);
		}
		score /= transitions.size();
		return score;
	}
	protected abstract double calcScoreFromTransition(ReusableModule<T> module, RelationManager<T> relMngr, TransitionMemory tm,
			ModelLearner modeler, T output, int outputKey);
	
//	protected void addOutputScore(T t, double score, int key) {
//		outputScores.add(new OutputScore<T>(t, score + (Math.random()-.5)/1000000, key));
//	}
	public void report() {
		System.out.println("Reporting on Module Manager");
		System.out.println(allModules.size() + " modules total");
		TreeSet<OutputScore<T>> ordered = new TreeSet<OutputScore<T>>(new Comparator<OutputScore<T>>() {
			@Override
			public int compare(OutputScore<T> arg0, OutputScore<T> arg1) {
				if (arg0.output instanceof Integer) return Integer.compare((int)arg0.output, (int)arg1.output);
				return arg0.output.toString().compareTo(arg1.output.toString());
			}
		});
		ordered.addAll(getDescOutputScores());
		for (OutputScore<T> n : ordered) {
			System.out.println(n.output + "	" + nodeModules.get(n.output) + "	score:	" + n.score);
		}
	}
	protected Collection<OutputScore<T>> getDescOutputScores() {
		throw new IllegalStateException("get rid of impure already");
	}
}
