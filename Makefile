black: FORCE
	black scratch checkmate experiments setup.py --line-length 127

test: FORCE
	pytest tests

FORCE: ;
