package ann;

import java.io.Serializable;
import java.util.*;

@SuppressWarnings("serial")
public class ExperienceReplay<T extends ExperienceReplay.Memory> implements Serializable {

	private final ArrayList<T> dataMemory = new ArrayList<T>();
	private final int maxSize;

	public ExperienceReplay(int maxSize) {
		this(maxSize, false);
	}
	// TODO overwrite duplicate experiences!
	public ExperienceReplay(int maxSize, boolean overwriteCopies) {
		this.maxSize = maxSize;
	}

	public void addExperience(ExperienceReplay<T> other) {
		for (T t : other.dataMemory) addMemory(t);
	}
	
	public void addMemory(T dp) {
		dataMemory.add(dp);
		if (dataMemory.size() > maxSize) dataMemory.remove(0);
	}

	public Collection<T> getBatch(int size, boolean resample) {
		if (size >= dataMemory.size()) return getBatch(); // efficienter
		Collection<T> result = new ArrayList<T>();
		Collection<T> replace = new ArrayList<T>();
		for (int i = 0; i < size; i++) {
			if (dataMemory.isEmpty()) return result;
			int k = (int) (dataMemory.size() * Math.random());
			T newDP = dataMemory.get(k);
			result.add(newDP);
			if (!resample) {
				dataMemory.remove(k);
				replace.add(newDP);
			}
		}
		dataMemory.addAll(replace);
		return result;
	}

	public Collection<T> getBatch() {
		Collections.shuffle(dataMemory);
		return dataMemory;
//		return getBatch(dataMemory.size(), false);
	}
	
	public boolean isFull() {
		return dataMemory.size() == maxSize;
	}

	public static interface Memory {
	}

	public void clear() {
		dataMemory.clear();
	}
	public double getSize() {
		return dataMemory.size();
	}
	public ArrayList<T> getInOrder() {
		return dataMemory;
	}
}
