package modulemanagement;

import java.util.TreeSet;

/**
 * stores the modules tested and their likelihood scores (for an output node)
 * low entropy means high faith in the highest score module (I think?)
 * one geographic strategy may be to learn RELs between nodes using the same modules,
 * then when a node with low entropy distribution has such a REL with a node with
 * low entropy distribution, it can "adopt" the distribution from its low entropy neighbor
 */
public class ModuleDistribution<T> {

	private final TreeSet<ModuleScore<T>> moduleScores = new TreeSet<ModuleScore<T>>();
	
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
	
	public double getHighestScore() {
		if (moduleScores.isEmpty()) return 0;
		return moduleScores.last().score;
	}
	
	@Override
	public String toString() {
		return getMostLikelyModule().toString();
	}
	
	private static class ModuleScore<T> implements Comparable<ModuleScore<T>> {
		
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
}
