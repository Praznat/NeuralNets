package ann.testing.display;

import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

import ann.testing.GridGame;

@SuppressWarnings("serial")
public class GridPanel extends JPanel {
	private final int gUnit;
	GridGame game;
	JFrame parent;
	public GridPanel(JFrame parent, int gUnit) {
		this.parent = parent;
		this.gUnit = gUnit;
	}
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		if (game != null) game.paintGrid(g);
	}
	public Dimension getPreferredSize() {
		if (game == null) return new Dimension(0, 0);
		return new Dimension(gUnit, gUnit);
	}
	public void setGame(GridGame game) {
		this.game = game;
		parent.pack();
	}
	public int getgUnit() {
		return gUnit;
	}
}