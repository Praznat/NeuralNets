package modulemanagement;

import java.util.TreeSet;

/**
 * stores the modules tested and their errors (for an output node)
 * high entropy means high faith in the lowest error module
 * one geographic strategy may be to learn RELs between nodes using the same modules,
 * then when a node with low entropy distribution has such a REL with a node with
 * high entropy distribution, it can "adopt" the distribution from its high entropy neighbor
 */
public class ModuleDistribution {

	private final TreeSet<ModuleError> moduleErrors = new TreeSet<ModuleError>();
	
	public ModuleDistribution() {}
	public ModuleDistribution(ReusableModule module, double error) {
		addModule(module, error);
	}
	
	public void addModule(ReusableModule module, double error) {
		moduleErrors.add(new ModuleError(module, error));
	}
	
	/**
	 * this is not probability entropy. this is error entropy. somehow it works.
	 * the higher this is, the MORE confidence in its best module (relative to other modules)
	 */
	public double getEntropy() {
		double sum = 0;
		for (ModuleError err : moduleErrors) sum += err.error * Math.log(err.error);
		return -sum;
	}

	public ReusableModule getMostLikelyModule() {
		return moduleErrors.first().module;
	}
	
	public double getLowestError() {
		return moduleErrors.first().error;
	}
	
	@Override
	public String toString() {
		return getMostLikelyModule().toString();
	}
	
	private static class ModuleError implements Comparable<ModuleError> {
		
		private ReusableModule module;
		private double error;

		public ModuleError(ReusableModule module, double error) {
			this.module = module;
			this.error = error;
		}

		@Override
		public int compareTo(ModuleError o) {
			return Double.compare(this.error, o.error);
		}
	}
}
