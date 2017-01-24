import os

"""
GitHub doesn't provide an include mechanism for README files so we have to
implement our own.
"""

def main():
  build_readme('README_raw.md', 'README.md')
  for d in os.listdir('.'):
    if not os.path.isdir(d) or d.startswith('.'):
      continue
    in_path = os.path.join(d, 'README_raw.md')
    out_path = os.path.join(d, 'README.md')
    build_readme(in_path, out_path)


def build_readme(in_path, out_path):
  if not os.path.isfile(in_path):
    return
  with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
    for line in fin:
      if not line.startswith(':INCLUDE'):
        fout.write('%s' % line)
      else:
        include_path = line.split(' ')[1].strip()
        include_path = os.path.join(os.path.split(in_path)[0], include_path)
        fout.write('# Code in file %s\n' % include_path)
        skip_toggle = False
        skip_next = False
        with open(include_path, 'r') as finc:
          for ll in finc:
            if ll.startswith('"""'):
              skip_next = skip_toggle
              skip_toggle = not skip_toggle
            elif not skip_toggle and not skip_next:
              fout.write(ll)
            elif skip_next:
              skip_next = False

if __name__ == '__main__':
  main()

