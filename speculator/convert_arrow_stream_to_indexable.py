import pyarrow as pa

filename = "data-00000-of-00003.arrow"
f = pa.ipc.open_stream(filename)
schema = pa.schema([pa.field('tokens', pa.uint32())])

fo = pa.ipc.new_file("indexable-"+filename, schema=schema)
for b in f:
    x = b['input_ids']
    for i in range(len(x)):
        doc = pa.record_batch([pa.array(x[i].as_py(), pa.uint32())], schema)
        fo.write(doc)
fo.close()
