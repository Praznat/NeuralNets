package modulemanagement;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;

import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modeler.TransitionMemory;

public abstract class ModuleManager<T> {

	protected final Map<T, ModuleDistribution<T>> nodeModules = new HashMap<T, ModuleDistribution<T>>();
	protected final Collection<ReusableModule<T>> allModules = new HashSet<ReusableModule<T>>();
	protected final TreeSet<OutputScore<T>> outputScores = new TreeSet<OutputScore<T>>();
	
	public abstract double[] getOutputs(ModelLearner modeler, RelationManager<T> relMngr, double[] stateVars, double[] action);
	public abstract void processFullModel(ModelLearner modeler, RelationManager<T> relMngr, int numTransitions);
	public void processFullModel(ModelLearner modeler, RelationManager<T> relMngr, int numTransitions, int times) {
		for (int i = 0; i < times; i++) processFullModel(modeler, relMngr, numTransitions);
	}
	
	public Map<T, ModuleDistribution<T>> getNodeModules() {
		return nodeModules;
	}
	
	protected ModuleDistribution<T> recalcModuleDistribution(ModuleDistribution<T> result, ModelLearner modeler,
			RelationManager<T> relMngr, T output, int outputKey, Collection<TransitionMemory> transitions) {
		if (allModules.isEmpty()) return result;
		result.clear();
		for (ReusableModule<T> module : allModules) {
			double score = calcScoresFromTransition(module, relMngr, transitions, modeler, output, outputKey);
			result.addModule(module, score);
		}
		return result;
	}
	protected double calcScoresFromTransition(ReusableModule<T> module, RelationManager<T> relMngr,
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
	
	protected void addOutputScore(T t, double score, int key) {
		outputScores.add(new OutputScore<T>(t, score + (Math.random()-.5)/1000000, key));
	}
	
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
		ordered.addAll(outputScores);
		for (OutputScore<T> n : ordered) {
			System.out.println(n.output + "	" + nodeModules.get(n.output) + "	score:	" + n.score);
		}
	}
}
