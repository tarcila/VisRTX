import csv
from pathlib import Path
import tempfile
import unittest

from analyze_fp64 import sass, source_locations, warnings


class AttributionTests(unittest.TestCase):
    def test_warning_occurrences_are_not_unique_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'log'
            path.write_text('Warning: Program is doing double precision computations. /x/a.h:12:3 (approximately)\n'*2
                            + 'Warning: Program is doing double precision computations. No source location available.\n')
            report = warnings(path)
            self.assertEqual(report['occurrences'], 3)
            self.assertEqual(report['files']['/x/a.h'], 2)
            self.assertEqual(report['locations']['/x/a.h:12'], 2)

    def test_executed_sass_counts_and_logical_local_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sass.csv'
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Kernel Name', 'callable'])
                writer.writerow(['Address', 'Source', '# Samples',
                                 'Predicated-On Thread Instructions Executed',
                                 'Address Space', 'Access Operation', 'Access Size'])
                writer.writerow(['0x10', '@P0 DADD.RM R0, R2, R4', 3, 7, '-', '-', '-'])
                writer.writerow(['0x20', 'LDL.64 R0, [R1]', 5, 11, 'Local', 'Load', 64])
                writer.writerow(['Kernel Name', 'duplicate'])
                writer.writerow(['0x10', '@P0 DADD.RM R0, R2, R4', 3, 7, '-', '-', '-'])
            result = sass(path, {'0x10': ('a.h', '1')})
            self.assertEqual(result['totals']['fp64_arithmetic_instructions'], 7)
            self.assertEqual(result['totals']['local_load_bytes'], 88)
            self.assertEqual(result['totals']['pc_samples'], 8)
            self.assertEqual(result['unique_sass_addresses'], 2)
            self.assertEqual(result['mapped_addresses'], 1)
            self.assertNotIn('duplicate', result['functions'])

    def test_first_correlated_location_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'correlated.csv'
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerows([
                    ['File Path', 'a.h'], ['4', 'code'], ['', '', '0x10', 'DADD'],
                    ['File Path', 'b.h'], ['9', 'inlined code'], ['', '', '0x10', 'DADD']])
            self.assertEqual(source_locations(path), {'0x10': ('a.h', '4')})


if __name__ == '__main__':
    unittest.main()
