  *	u?Vrf@2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorkGq?::??!??̖?,8@)kGq?::??1??̖?,8@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate3p@KW???!?aU\ ?7@)R<??k??1??s?2?1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeati??I??!A?F
??D@)???c????1 a?}<@1@:Preprocessing2T
Iterator::Root::ParallelMapV232?]?)??!b?S;?t,@)32?]?)??1b?S;?t,@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip9?]????!?'?4? S@)L?
F%u??1?2a|M$@:Preprocessing2E
Iterator::Root]p????!?aW-W}7@)??D????1g-[!?"@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?4f??!?S?S7?@)?4f??1?S?S7?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??5"??!K???9@)?KTole?1??K9?L??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.