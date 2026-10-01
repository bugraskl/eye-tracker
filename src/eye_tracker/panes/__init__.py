"""Split-pane focus (experimental): keyboard focus follows the gaze between the
split panes of the focused terminal window.

* :mod:`.types` - panes, snapshots and the provider protocol,
* :mod:`.registry` - which providers look at which windows, and the deny-list,
* :mod:`.providers` - tmux and WezTerm, through their own command-line tools,
  Windows Terminal and (opt-in) the side-by-side sessions of the Claude and
  ChatGPT desktop apps,
  through UI Automation,
* :mod:`.worker` - runs providers on a background thread,
* :mod:`.decider` - pure logic that decides when to move the focus.

No input is ever synthesised: providers ask the terminal itself to focus a pane.
"""
