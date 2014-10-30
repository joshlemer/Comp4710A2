import apriorialg as ap

a = ap.load_dataset()
b,c = ap.apriori(a, 0.002)
d = ap.generateRules(b,c, 0.002)

f = ap.filter_by_lift(c,d)
f = ap.filter_by_interest(c,f)
f = ap.filter_by_ps(c,f, 0.0035)
f = ap.filter_by_phi(c,f, 0.35)
ap.print_rules(f)


