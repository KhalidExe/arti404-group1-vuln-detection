# Error candidates (seed 42, original test set)

Most confident wrong predictions per group. Probabilities are P(vulnerable).

## Both models wrong

- id **4645**, true label Vulnerable (1), 181 CodeBERT tokens
  - TF-IDF + RF: original pred 0 (p=0.217), renamed pred 0 (p=0.467)
  - CodeBERT: original pred 0 (p=0.003), renamed pred 0 (p=0.062)

```c
build_tpm2(GArray *table_data, BIOSLinker *linker) { Acpi20TPM2 *tpm2_ptr; tpm2_ptr = acpi_data_push(table_data, sizeof *tpm2_ptr); tpm2_ptr->platform_class = cpu_to_le16(TPM2_ACPI_CLASS_CLIENT); tpm2_ptr->control_area_address = cpu_to_le64(0); tpm2_ptr->start_method = cpu_to_le32(TPM2_START_METHOD_MMIO); build_header(linker, table_data, (void *)tpm2_ptr, "TPM2", sizeof(*tpm2_ptr), 4, NULL, NULL); }
```

- id **669**, true label Vulnerable (1), 1262 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 0 (p=0.213), renamed pred 0 (p=0.349)
  - CodeBERT: original pred 0 (p=0.007), renamed pred 1 (p=0.588)

```c
static void do_interrupt64(CPUX86State *env, int intno, int is_int, int error_code, target_ulong next_eip, int is_hw) { SegmentCache *dt; target_ulong ptr; int type, dpl, selector, cpl, ist; int has_error_code, new_stack; uint32_t e1, e2, e3, ss; target_ulong old_eip, esp, offset; has_error_code = 0; if (!is_int && !is_hw) { has_error_code = exception_has_error_code(intno); } if (is_int) { old_eip = next_eip; } else { old_eip = env->eip; } dt = &env->idt; if (intno * 16 + 15 > dt->limit) { raise_exception_err(env, EXCP0D_GPF, intno * 16 + 2); } ptr = dt->base + intno * 16; e1 = cpu_ldl_kernel(env, ptr); e2 = cpu_ldl_kernel(env, ptr + 4); e3 = cpu_ldl_kernel(env, ptr + 8); type = (e2 >> DESC_ ...
```

  Renamed version:

```c
static void FUN1(CPUX86State *VAR1, int VAR2, int VAR3, int VAR4, target_ulong VAR5, int VAR6) { SegmentCache *VAR7; target_ulong VAR8; int VAR9, VAR10, VAR11, VAR12, VAR13; int VAR14, VAR15; uint32_t VAR16, VAR17, VAR18, VAR19; target_ulong VAR20, VAR21, VAR22; VAR14 = 0; if (!VAR3 && !VAR6) { VAR14 = FUN2(VAR2); } if (VAR3) { VAR20 = VAR5; } else { VAR20 = VAR1->eip; } VAR7 = &VAR1->idt; if (VAR2 * 16 + 15 > VAR7->limit) { FUN3(VAR1, EXCP0D_GPF, VAR2 * 16 + 2); } VAR8 = VAR7->base + VAR2 * 16; VAR16 = FUN4(VAR1, VAR8); VAR17 = FUN4(VAR1, VAR8 + 4); VAR18 = FUN4(VAR1, VAR8 + 8); VAR9 = (VAR17 >> DESC_TYPE_SHIFT) & 0x1f; switch (VAR9) { case 14: case 15: break; default: FUN3(VAR1, EXCP0D_GPF ...
```

- id **14248**, true label Vulnerable (1), 2445 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 0 (p=0.286), renamed pred 0 (p=0.288)
  - CodeBERT: original pred 0 (p=0.003), renamed pred 0 (p=0.111)

```c
LF_FUNC (h, luma, sse2) LF_IFUNC(h, luma_intra, sse2) LF_FUNC (v, luma, sse2) LF_IFUNC(v, luma_intra, sse2) #define H264_WEIGHT(W, H, OPT) \ void ff_h264_weight_ ## W ## x ## H ## _ ## OPT(uint8_t *dst, \ int stride, int log2_denom, int weight, int offset); #define H264_BIWEIGHT(W, H, OPT) \ void ff_h264_biweight_ ## W ## x ## H ## _ ## OPT(uint8_t *dst, \ uint8_t *src, int stride, int log2_denom, int weightd, \ int weights, int offset); #define H264_BIWEIGHT_MMX(W,H) \ H264_WEIGHT (W, H, mmx2) \ H264_BIWEIGHT(W, H, mmx2) #define H264_BIWEIGHT_MMX_SSE(W,H) \ H264_BIWEIGHT_MMX(W, H) \ H264_WEIGHT (W, H, sse2) \ H264_BIWEIGHT (W, H, sse2) \ H264_BIWEIGHT (W, H, ssse3) H264_BIWEIGHT_MMX_SSE(16, ...
```

## Only TF-IDF + RF wrong

- id **7448**, true label Safe (0), 1589 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 1 (p=0.761), renamed pred 1 (p=0.660)
  - CodeBERT: original pred 0 (p=0.391), renamed pred 1 (p=0.594)

```c
static int encode_apng(AVCodecContext *avctx, AVPacket *pkt, const AVFrame *pict, int *got_packet) { PNGEncContext *s = avctx->priv_data; int ret; int enc_row_size; size_t max_packet_size; APNGFctlChunk fctl_chunk = {0}; if (pict && avctx->codec_id == AV_CODEC_ID_APNG && s->color_type == PNG_COLOR_TYPE_PALETTE) { uint32_t checksum = ~av_crc(av_crc_get_table(AV_CRC_32_IEEE_LE), ~0U, pict->data[1], 256 * sizeof(uint32_t)); if (avctx->frame_number == 0) { s->palette_checksum = checksum; } else if (checksum != s->palette_checksum) { av_log(avctx, AV_LOG_ERROR, "Input contains more than one unique palette. APNG does not support multiple palettes.\n"); return -1; } } enc_row_size = deflateBound(&s ...
```

  Renamed version:

```c
static int FUN1(AVCodecContext *VAR1, AVPacket *VAR2, const AVFrame *VAR3, int *VAR4) { PNGEncContext *VAR5 = VAR1->priv_data; int VAR6; int VAR7; size_t VAR8; APNGFctlChunk VAR9 = {0}; if (VAR3 && VAR1->codec_id == AV_CODEC_ID_APNG && VAR5->color_type == PNG_COLOR_TYPE_PALETTE) { uint32_t VAR10 = ~FUN2(FUN3(AV_CRC_32_IEEE_LE), ~0U, VAR3->data[1], 256 * sizeof(uint32_t)); if (VAR1->frame_number == 0) { VAR5->palette_checksum = VAR10; } else if (VAR10 != VAR5->palette_checksum) { FUN4(VAR1, AV_LOG_ERROR, "Input contains more than one unique palette. APNG does not support multiple palettes.\n"); return -1; } } VAR7 = FUN5(&VAR5->zstream, (VAR1->width * VAR5->bits_per_pixel + 7) >> 3); VAR8 = A ...
```

- id **5356**, true label Safe (0), 3416 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 1 (p=0.757), renamed pred 1 (p=0.644)
  - CodeBERT: original pred 0 (p=0.369), renamed pred 0 (p=0.216)

```c
av_cold int MPV_common_init(MpegEncContext *s) { int y_size, c_size, yc_size, i, mb_array_size, mv_table_size, x, y, threads; if(s->codec_id == CODEC_ID_MPEG2VIDEO && !s->progressive_sequence) s->mb_height = (s->height + 31) / 32 * 2; else if (s->codec_id != CODEC_ID_H264) s->mb_height = (s->height + 15) / 16; if(s->avctx->pix_fmt == PIX_FMT_NONE){ av_log(s->avctx, AV_LOG_ERROR, "decoding to PIX_FMT_NONE is not supported.\n"); return -1; } if((s->encoding || (s->avctx->active_thread_type & FF_THREAD_SLICE)) && (s->avctx->thread_count > MAX_THREADS || (s->avctx->thread_count > s->mb_height && s->mb_height))){ av_log(s->avctx, AV_LOG_ERROR, "too many threads\n"); return -1; } if((s->width || s ...
```

## Only CodeBERT wrong

- id **3076**, true label Vulnerable (1), 229 CodeBERT tokens
  - TF-IDF + RF: original pred 1 (p=0.520), renamed pred 1 (p=0.523)
  - CodeBERT: original pred 0 (p=0.004), renamed pred 0 (p=0.138)

```c
void qemu_spice_create_host_primary(SimpleSpiceDisplay *ssd) { QXLDevSurfaceCreate surface; memset(&surface, 0, sizeof(surface)); dprint(1, "%s/%d: %dx%d\n", __func__, ssd->qxl.id, surface_width(ssd->ds), surface_height(ssd->ds)); surface.format = SPICE_SURFACE_FMT_32_xRGB; surface.width = surface_width(ssd->ds); surface.height = surface_height(ssd->ds); surface.stride = -surface.width * 4; surface.mouse_mode = true; surface.flags = 0; surface.type = 0; surface.mem = (uintptr_t)ssd->buf; surface.group_id = MEMSLOT_GROUP_HOST; qemu_spice_create_primary_surface(ssd, 0, &surface, QXL_SYNC); }
```

- id **6272**, true label Vulnerable (1), 246 CodeBERT tokens
  - TF-IDF + RF: original pred 1 (p=0.577), renamed pred 1 (p=0.580)
  - CodeBERT: original pred 0 (p=0.005), renamed pred 1 (p=0.795)

```c
static void lpc_analyze_remodulate(int32_t *decoded, const int coeffs[32], int order, int qlevel, int len, int bps) { int i, j; int ebps = 1 << (bps-1); unsigned sigma = 0; for (i = order; i < len; i++) sigma |= decoded[i] + ebps; if (sigma < 2*ebps) return; for (i = len - 1; i >= order; i--) { int64_t p = 0; for (j = 0; j < order; j++) p += coeffs[j] * (int64_t)decoded[i-order+j]; decoded[i] -= p >> qlevel; } for (i = order; i < len; i++, decoded++) { int32_t p = 0; for (j = 0; j < order; j++) p += coeffs[j] * (uint32_t)decoded[j]; decoded[j] += p >> qlevel; } }
```

  Renamed version:

```c
static void FUN1(int32_t *VAR1, const int VAR2[32], int VAR3, int VAR4, int VAR5, int VAR6) { int VAR7, VAR8; int VAR9 = 1 << (VAR6-1); unsigned VAR10 = 0; for (VAR7 = VAR3; VAR7 < VAR5; VAR7++) VAR10 |= VAR1[VAR7] + VAR9; if (VAR10 < 2*VAR9) return; for (VAR7 = VAR5 - 1; VAR7 >= VAR3; VAR7--) { int64_t VAR11 = 0; for (VAR8 = 0; VAR8 < VAR3; VAR8++) VAR11 += VAR2[VAR8] * (int64_t)VAR1[VAR7-VAR3+VAR8]; VAR1[VAR7] -= VAR11 >> VAR4; } for (VAR7 = VAR3; VAR7 < VAR5; VAR7++, VAR1++) { int32_t VAR11 = 0; for (VAR8 = 0; VAR8 < VAR3; VAR8++) VAR11 += VAR2[VAR8] * (uint32_t)VAR1[VAR8]; VAR1[VAR8] += VAR11 >> VAR4; } }
```

## Truncated (> 512 tokens), CodeBERT wrong

- id **19557**, true label Vulnerable (1), 3197 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 0 (p=0.325), renamed pred 0 (p=0.322)
  - CodeBERT: original pred 0 (p=0.004), renamed pred 0 (p=0.018)

```c
DECLARE_LOOP_FILTER(mmxext) DECLARE_LOOP_FILTER(sse2) DECLARE_LOOP_FILTER(ssse3) DECLARE_LOOP_FILTER(sse4) #endif #define VP8_LUMA_MC_FUNC(IDX, SIZE, OPT) \ c->put_vp8_epel_pixels_tab[IDX][0][2] = ff_put_vp8_epel ## SIZE ## _h6_ ## OPT; \ c->put_vp8_epel_pixels_tab[IDX][2][0] = ff_put_vp8_epel ## SIZE ## _v6_ ## OPT; \ c->put_vp8_epel_pixels_tab[IDX][2][2] = ff_put_vp8_epel ## SIZE ## _h6v6_ ## OPT #define VP8_MC_FUNC(IDX, SIZE, OPT) \ c->put_vp8_epel_pixels_tab[IDX][0][1] = ff_put_vp8_epel ## SIZE ## _h4_ ## OPT; \ c->put_vp8_epel_pixels_tab[IDX][1][0] = ff_put_vp8_epel ## SIZE ## _v4_ ## OPT; \ c->put_vp8_epel_pixels_tab[IDX][1][1] = ff_put_vp8_epel ## SIZE ## _h4v4_ ## OPT; \ c->put_vp8_e ...
```

- id **23779**, true label Vulnerable (1), 1150 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 0 (p=0.420), renamed pred 0 (p=0.414)
  - CodeBERT: original pred 0 (p=0.005), renamed pred 0 (p=0.194)

```c
int ppc_hash64_handle_mmu_fault(PowerPCCPU *cpu, target_ulong eaddr, int rwx, int mmu_idx) { CPUState *cs = CPU(cpu); CPUPPCState *env = &cpu->env; ppc_slb_t *slb; hwaddr pte_offset; ppc_hash_pte64_t pte; int pp_prot, amr_prot, prot; uint64_t new_pte1; const int need_prot[] = {PAGE_READ, PAGE_WRITE, PAGE_EXEC}; hwaddr raddr; assert((rwx == 0) || (rwx == 1) || (rwx == 2)); if (((rwx == 2) && (msr_ir == 0)) || ((rwx != 2) && (msr_dr == 0))) { raddr = eaddr & 0x0FFFFFFFFFFFFFFFULL; tlb_set_page(cs, eaddr & TARGET_PAGE_MASK, raddr & TARGET_PAGE_MASK, PAGE_READ | PAGE_WRITE | PAGE_EXEC, mmu_idx, TARGET_PAGE_SIZE); return 0; } slb = slb_lookup(cpu, eaddr); if (!slb) { if (rwx == 2) { cs->exception ...
```

## Prediction flips after renaming (and is wrong in one version)

- id **15582**, true label Vulnerable (1), 73 CodeBERT tokens
  - TF-IDF + RF: original pred 0 (p=0.328), renamed pred 1 (p=0.504)
  - CodeBERT: original pred 0 (p=0.009), renamed pred 0 (p=0.065)

```c
static void qmp_output_end_struct(Visitor *v, Error **errp) { QmpOutputVisitor *qov = to_qov(v); QObject *value = qmp_output_pop(qov); assert(qobject_type(value) == QTYPE_QDICT); }
```

  Renamed version:

```c
static void FUN1(Visitor *VAR1, Error **VAR2) { QmpOutputVisitor *VAR3 = FUN2(VAR1); QObject *VAR4 = FUN3(VAR3); assert(FUN4(VAR4) == QTYPE_QDICT); }
```

- id **16708**, true label Vulnerable (1), 565 CodeBERT tokens (truncated)
  - TF-IDF + RF: original pred 0 (p=0.321), renamed pred 0 (p=0.382)
  - CodeBERT: original pred 0 (p=0.025), renamed pred 1 (p=0.852)

```c
static uint32_t hpet_ram_readl(void *opaque, target_phys_addr_t addr) { HPETState *s = (HPETState *)opaque; uint64_t cur_tick, index; DPRINTF("qemu: Enter hpet_ram_readl at %" PRIx64 "\n", addr); index = addr; if (index >= 0x100 && index <= 0x3ff) { uint8_t timer_id = (addr - 0x100) / 0x20; if (timer_id > HPET_NUM_TIMERS - 1) { printf("qemu: timer id out of range\n"); return 0; } HPETTimer *timer = &s->timer[timer_id]; switch ((addr - 0x100) % 0x20) { case HPET_TN_CFG: return timer->config; case HPET_TN_CFG + 4: return timer->config >> 32; case HPET_TN_CMP: return timer->cmp; case HPET_TN_CMP + 4: return timer->cmp >> 32; case HPET_TN_ROUTE: return timer->fsb >> 32; default: DPRINTF("qemu: i ...
```

  Renamed version:

```c
static uint32_t FUN1(void *VAR1, target_phys_addr_t VAR2) { HPETState *VAR3 = (HPETState *)VAR1; uint64_t VAR4, index; DPRINTF("qemu: Enter hpet_ram_readl at %" VAR5 "\n", VAR2); index = VAR2; if (index >= 0x100 && index <= 0x3ff) { uint8_t VAR6 = (VAR2 - 0x100) / 0x20; if (VAR6 > HPET_NUM_TIMERS - 1) { printf("qemu: timer id out of range\n"); return 0; } HPETTimer *VAR7 = &VAR3->timer[VAR6]; switch ((VAR2 - 0x100) % 0x20) { case HPET_TN_CFG: return VAR7->config; case HPET_TN_CFG + 4: return VAR7->config >> 32; case HPET_TN_CMP: return VAR7->cmp; case HPET_TN_CMP + 4: return VAR7->cmp >> 32; case HPET_TN_ROUTE: return VAR7->fsb >> 32; default: DPRINTF("qemu: invalid hpet_ram_readl\n"); break ...
```

- id **18195**, true label Vulnerable (1), 246 CodeBERT tokens
  - TF-IDF + RF: original pred 0 (p=0.370), renamed pred 1 (p=0.511)
  - CodeBERT: original pred 0 (p=0.006), renamed pred 0 (p=0.029)

```c
static void test_dispatch_cmd_failure(void) { QDict *req = qdict_new(); QObject *resp; qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd2"))); resp = qmp_dispatch(QOBJECT(req)); assert(resp != NULL); assert(qdict_haskey(qobject_to_qdict(resp), "error")); qobject_decref(resp); QDECREF(req); req = qdict_new(); qdict_put(args, "a", qint_from_int(66)); qdict_put(req, "arguments", args); qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd"))); resp = qmp_dispatch(QOBJECT(req)); assert(resp != NULL); assert(qdict_haskey(qobject_to_qdict(resp), "error")); qobject_decref(resp); QDECREF(req); }
```

  Renamed version:

```c
static void FUN1(void) { QDict *VAR1 = FUN2(); QObject *VAR2; FUN3(VAR1, "execute", QOBJECT(FUN4("user_def_cmd2"))); VAR2 = FUN5(QOBJECT(VAR1)); assert(VAR2 != NULL); assert(FUN6(FUN7(VAR2), "error")); FUN8(VAR2); QDECREF(VAR1); VAR1 = FUN2(); FUN9(VAR3, "a", FUN10(66)); FUN9(VAR1, "arguments", VAR3); FUN3(VAR1, "execute", QOBJECT(FUN4("user_def_cmd"))); VAR2 = FUN5(QOBJECT(VAR1)); assert(VAR2 != NULL); assert(FUN6(FUN7(VAR2), "error")); FUN8(VAR2); QDECREF(VAR1); }
```
