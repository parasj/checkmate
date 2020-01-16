black: FORCE
	black scripts checkmate setup.py --line-length 127

test: FORCE
	pytest tests

FORCE: ;
