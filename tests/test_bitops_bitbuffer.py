
from mhdm.bitops import BitBuffer


def test_write_read_buffer():
	buffer = BitBuffer()
	buffer.write(42, 8)
	buffer.write(7, 8)
	read = buffer.read(8)
	assert(read == 42)
	read = buffer.read(8)
	assert(read == 7)


def test_write_read_file(tmpdir):
	p = tmpdir.join('~test_write_read_file.bin')
	buffer = BitBuffer(p, 'wb')
	buffer.write(0x42, 8)
	buffer.write(7, 4)
	assert(buffer.buffer == 0xFF427)
	buffer.open(p, 'rb')
	read = buffer.read(8)
	assert(read == 0x42)
	read = buffer.read(4)
	assert(read == 7)
