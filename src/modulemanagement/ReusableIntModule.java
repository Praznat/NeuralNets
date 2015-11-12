package modulemanagement;

import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;

@SuppressWarnings("serial")
public class ReusableIntModule extends ReusableModule<Integer> {

	/**
	 * creates a reusable module
	 * @param relMngr 
	 */
	public static ReusableIntModule createNeighborHoodModule(ModelLearner modeler,
			RelationManager<Integer> relMngr, int outputOfInterest, int[] numHidden,
			int epochs, double lRate, double mRate, double sRate) {
		ReusableIntModule result = new ReusableIntModule();
		result.relations.addAll(relMngr.getUsedRels());
		result.trainModuleNN(modeler.getExperience(), outputOfInterest, relMngr, numHidden,
				epochs, lRate, mRate, sRate);
		return result;
	}
	
	@Override
	protected int getVectorKey(Integer key) {
		return key != null ? key : -1;
	}

}
