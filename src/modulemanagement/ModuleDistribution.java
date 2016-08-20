package modulemanagement;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;

import ann.Utils;
import ann.indirectencodings.IndirectInput;
import reasoner.DiscreteState;

/**
 * stores the modules tested and their likelihood scores (for an output node)
 * low entropy means high faith in the highest score module (I think?)
 * one geographic strategy may be to learn RELs between nodes using the same modules,
 * then when a node with low entropy distribution has such a REL with a node with
 * low entropy distribution, it can "adopt" the distribution from its low entropy neighbor
 */
@SuppressWarnings("serial")
public class ModuleDistribution<T> implements Serializable {

	private final TreeSet<ModuleScore<T>> moduleScores = new TreeSet<ModuleScore<T>>();
	private final Map<DiscreteState, Double> data = new HashMap<DiscreteState, Double>();
	
	public ModuleDistribution() {}
	public ModuleDistribution(ReusableModule<T> module, double score) {
		addModule(module, score);
	}
	
	public void addModule(ReusableModule<T> module, double score) {
		moduleScores.add(new ModuleScore<T>(module, score));
	}
	
	// because score should basically be likelihood
	public double getEntropy() {
		double sum = 0;
		for (ModuleScore<T> ms : moduleScores) sum += ms.score * Math.log(ms.score);
		return -sum;
	}

	public ReusableModule<T> getMostLikelyModule() {
		if (moduleScores.isEmpty()) return null;
		return moduleScores.last().module;
	}
	public ReusableModule<T> drawModuleProbabilistically() {
		// TODO Auto-generated method stub
		return null;
	}
	public Collection<ReusableModule<T>> getModulesAboveThresh(double minScore) {
		Collection<ReusableModule<T>> result = new ArrayList<ReusableModule<T>>();
		for (Iterator<ModuleScore<T>>iter = moduleScores.descendingIterator(); iter.hasNext();) {
			ModuleScore<T> ms = iter.next();
			if (ms.score >= minScore) result.add(ms.module);
			else break;
		}
		return result;
	}

	public void removeModule(ReusableModule<Integer> module) {
		for (Iterator<ModuleScore<T>> iter = moduleScores.iterator(); iter.hasNext();) {
			ModuleScore<T> ms = iter.next();
			if (ms.module == module) {
				iter.remove();
				break;
			}
		}
	}
	
	public double getHighestScore() {
		if (moduleScores.isEmpty()) return 0;
		return moduleScores.last().score;
	}
	
	public Map<DiscreteState, Double> getData() {
		return data;
	}
	
	@Override
	public String toString() {
		return getMostLikelyModule().toString();
	}
	
	public boolean isEmpty() {
		return moduleScores.isEmpty();
	}
	public void clear() {
		moduleScores.clear();
	}
	public int size() {
		return moduleScores.size();
	}

	public void report(int max) {
		int i = 0;
		for (Iterator<ModuleScore<T>>iter = moduleScores.descendingIterator(); iter.hasNext();) {
			if (i++ > max) break;
			System.out.println(iter.next());
		}
		System.out.println();
		ReusableModule<T> mmm = getMostLikelyModule();
		ArrayList<IndirectInput> relations = mmm.getRelations();
		System.out.println("Freq1	EstP1	Input State");
		for (Map.Entry<DiscreteState, Double> entry : data.entrySet()) {
			System.out.println(Utils.round(entry.getValue(), 2)
					+ "	" + Utils.round(mmm.getNNOutput(entry.getKey().getRawState()),2)
					+ " " + discreteStateToRels(entry.getKey(), relations));
		}
	}
	
	public static String discreteStateToRels(DiscreteState ds, ArrayList<IndirectInput> relations) {
		double[] rawState = ds.getRawState();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < rawState.length; i++) 
			if (rawState[i] > 0) sb.append(relations.get(i)).append(".");
		return sb.toString();
	}
	
	private static class ModuleScore<T> implements Comparable<ModuleScore<T>>, Serializable {
		
		private ReusableModule<T> module;
		private double score;

		public ModuleScore(ReusableModule<T> module, double score) {
			this.module = module;
			this.score = score;
		}

		@Override
		public int compareTo(ModuleScore<T> o) {
			return Double.compare(this.score, o.score);
		}
		
		@Override
		public String toString() {
			return score + "	" + module;
		}
	}
}
