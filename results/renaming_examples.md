# Renaming examples (real test set)

Same function before and after `rename_identifiers`. The label does not change.

## id 18384, label Vulnerable (1)

Renamed: 7 function name(s), 5 variable name(s). Parse errors before/after: 0/0.

Before:

```c
static void blk_send(QEMUFile *f, BlkMigBlock * blk) { int len; uint64_t flags = BLK_MIG_FLAG_DEVICE_BLOCK; if (block_mig_state.zero_blocks && buffer_is_zero(blk->buf, BLOCK_SIZE)) { flags |= BLK_MIG_FLAG_ZERO_BLOCK; } qemu_put_be64(f, (blk->sector << BDRV_SECTOR_BITS) | flags); len = strlen(bdrv_get_device_name(blk->bmds->bs)); qemu_put_byte(f, len); qemu_put_buffer(f, (uint8_t *)bdrv_get_device_name(blk->bmds->bs), len); if (flags & BLK_MIG_FLAG_ZERO_BLOCK) { qemu_fflush(f); return; } qemu_put_buffer(f, blk->buf, BLOCK_SIZE); }
```

After:

```c
static void FUN1(QEMUFile *VAR1, BlkMigBlock * VAR2) { int VAR3; uint64_t VAR4 = BLK_MIG_FLAG_DEVICE_BLOCK; if (VAR5.zero_blocks && FUN2(VAR2->buf, BLOCK_SIZE)) { VAR4 |= BLK_MIG_FLAG_ZERO_BLOCK; } FUN3(VAR1, (VAR2->sector << BDRV_SECTOR_BITS) | VAR4); VAR3 = strlen(FUN4(VAR2->bmds->bs)); FUN5(VAR1, VAR3); FUN6(VAR1, (uint8_t *)FUN4(VAR2->bmds->bs), VAR3); if (VAR4 & BLK_MIG_FLAG_ZERO_BLOCK) { FUN7(VAR1); return; } FUN6(VAR1, VAR2->buf, BLOCK_SIZE); }
```

## id 3484, label Vulnerable (1)

Renamed: 3 function name(s), 5 variable name(s). Parse errors before/after: 0/0.

Before:

```c
static uint64_t icp_pit_read(void *opaque, target_phys_addr_t offset, unsigned size) { icp_pit_state *s = (icp_pit_state *)opaque; int n; n = offset >> 8; if (n > 3) { hw_error("sp804_read: Bad timer %d\n", n); } return arm_timer_read(s->timer[n], offset & 0xff); }
```

After:

```c
static uint64_t FUN1(void *VAR1, target_phys_addr_t VAR2, unsigned VAR3) { icp_pit_state *VAR4 = (icp_pit_state *)VAR1; int VAR5; VAR5 = VAR2 >> 8; if (VAR5 > 3) { FUN2("sp804_read: Bad timer %d\n", VAR5); } return FUN3(VAR4->timer[VAR5], VAR2 & 0xff); }
```

## id 24123, label Vulnerable (1)

Renamed: 3 function name(s), 3 variable name(s). Parse errors before/after: 0/0.

Before:

```c
static void ehci_trace_qtd(EHCIQueue *q, target_phys_addr_t addr, EHCIqtd *qtd) { trace_usb_ehci_qtd(q, addr, qtd->next, qtd->altnext, get_field(qtd->token, QTD_TOKEN_TBYTES), get_field(qtd->token, QTD_TOKEN_CPAGE), get_field(qtd->token, QTD_TOKEN_CERR), get_field(qtd->token, QTD_TOKEN_PID), (bool)(qtd->token & QTD_TOKEN_IOC), (bool)(qtd->token & QTD_TOKEN_ACTIVE), (bool)(qtd->token & QTD_TOKEN_HALT), (bool)(qtd->token & QTD_TOKEN_BABBLE), (bool)(qtd->token & QTD_TOKEN_XACTERR)); }
```

After:

```c
static void FUN1(EHCIQueue *VAR1, target_phys_addr_t VAR2, EHCIqtd *VAR3) { FUN2(VAR1, VAR2, VAR3->next, VAR3->altnext, FUN3(VAR3->token, QTD_TOKEN_TBYTES), FUN3(VAR3->token, QTD_TOKEN_CPAGE), FUN3(VAR3->token, QTD_TOKEN_CERR), FUN3(VAR3->token, QTD_TOKEN_PID), (bool)(VAR3->token & QTD_TOKEN_IOC), (bool)(VAR3->token & QTD_TOKEN_ACTIVE), (bool)(VAR3->token & QTD_TOKEN_HALT), (bool)(VAR3->token & QTD_TOKEN_BABBLE), (bool)(VAR3->token & QTD_TOKEN_XACTERR)); }
```

## id 23555, label Safe (0)

Renamed: 2 function name(s), 4 variable name(s). Parse errors before/after: 0/0.

Before:

```c
static void s390_flic_common_realize(DeviceState *dev, Error **errp) { S390FLICState *fs = S390_FLIC_COMMON(dev); uint32_t max_batch = fs->adapter_routes_max_batch; if (max_batch > ADAPTER_ROUTES_MAX_GSI) { error_setg(errp, "flic property adapter_routes_max_batch too big" " (%d > %d)", max_batch, ADAPTER_ROUTES_MAX_GSI); } fs->ais_supported = true; }
```

After:

```c
static void FUN1(DeviceState *VAR1, Error **VAR2) { S390FLICState *VAR3 = S390_FLIC_COMMON(VAR1); uint32_t VAR4 = VAR3->adapter_routes_max_batch; if (VAR4 > ADAPTER_ROUTES_MAX_GSI) { FUN2(VAR2, "flic property adapter_routes_max_batch too big" " (%d > %d)", VAR4, ADAPTER_ROUTES_MAX_GSI); } VAR3->ais_supported = true; }
```

## id 16415, label Safe (0)

Renamed: 4 function name(s), 8 variable name(s). Parse errors before/after: 0/0.

Before:

```c
static void pc_dimm_get_size(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) { int64_t value; MemoryRegion *mr; PCDIMMDevice *dimm = PC_DIMM(obj); mr = host_memory_backend_get_memory(dimm->hostmem, errp); value = memory_region_size(mr); visit_type_int(v, name, &value, errp); }
```

After:

```c
static void FUN1(Object *VAR1, Visitor *VAR2, const char *VAR3, void *VAR4, Error **VAR5) { int64_t VAR6; MemoryRegion *VAR7; PCDIMMDevice *VAR8 = PC_DIMM(VAR1); VAR7 = FUN2(VAR8->hostmem, VAR5); VAR6 = FUN3(VAR7); FUN4(VAR2, VAR3, &VAR6, VAR5); }
```
